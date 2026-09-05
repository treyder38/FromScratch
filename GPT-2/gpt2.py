import os
import math 
import time
import torch 
import inspect
import tiktoken
import numpy as np
import torch.nn as nn
import torch.distributed
import json, logging, datetime
from dataclasses import dataclass
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class Logger:
    def __init__(self, log_dir, run_name, enabled=True):
        self.enabled = enabled
        if not enabled:
            return
        self.run_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.config_path = os.path.join(self.run_dir, "config.json")
        self.metrics = open(os.path.join(self.run_dir, "train.jsonl"), "a")

        self.log = logging.getLogger(run_name)
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        self.log.addHandler(handler)

    def set_config(self, config):
        if not self.enabled:
            return
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        self.info(f"config saved to {self.config_path}")

    def metric(self, **kw):
        if not self.enabled:
            return
        self.metrics.write(json.dumps(kw) + "\n")
        self.metrics.flush()
        self.info(" | ".join(f"{k} {v:.5g}" if isinstance(v, float) else f"{k} {v}"
                             for k, v in kw.items()))

    def info(self, msg):
        if self.enabled:
            self.log.info(msg)

    def close(self):
        if self.enabled:
            self.metrics.close()

# -----------------------------------

class MLP(nn.Sequential):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_emb, 4 * config.n_emb)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_emb, config.n_emb)
        self.c_proj.NANOGPT_SCALE_INIT = 1


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_emb % config.n_head == 0

        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb)

        # output projection
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        self.n_head = config.n_head
        self.n_emb = config.n_emb

        # bias in HF implementation = mask
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                      .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x) # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_emb, dim=2) # (B, T, C) 3 times
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))     # (B, n_head, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # (B, n_head, T, T)
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # Pre Norm
        x = x + self.mlp(self.ln_2(x))
        return x

# -----------------------------------

def load_tokens(filename):
    npt = np.fromfile(filename, dtype=np.uint16).astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)

class DataLoaderLite:

    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = sorted([s for s in os.listdir(data_root) if split in s])
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        assert len(self.shards) > 0, f"no shards found for split {split}" 
        logger.info(f"found {len(self.shards)} shards for split {split}")

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y

# -----------------------------------


@dataclass
class GPTconfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_emb: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_emb),
                wpe = nn.Embedding(config.block_size, config.n_emb),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_emb)
            )
        )
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)

        # weights tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):

            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn. Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def configure_optimizer(self, weight_decay, lr, device):
        params_dict = {p_name : p for p_name, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for p in params_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in params_dict.values() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params   = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} params")
        logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,}, params")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in str(device)
        logger.info(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"model's input can't be more than {self.config.block_size} tokens"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        pos_emb = self.transformer.wpe(pos) # (T, n_emb)
        tok_emb = self.transformer.wte(idx) # (B, T, n_emb)

        x = tok_emb + pos_emb # (B, T, n_emb)
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x) 
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, logits.size(-1)), targets.view(B * T))

        return logits, loss


# -----------------------------------


if __name__ == "__main__":

    # set up DDP
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), f"cuda is required for ddp"
        init_process_group(backend='nccl')
        ddp_rank       = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
    else:
        ddp_rank       = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

    # setup logger
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "log"
    logger = Logger(log_dir, run_name, enabled=master_process)
    logger.info(f"using device: {device}")

    # reproducibility
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # precision
    torch.set_float32_matmul_precision('high')

    # batch size setup
    batch_size = 524288  # ~0.5M like in GPT3 small
    B, T = 64, 1024      # micro batch size & context length
    assert batch_size % (B * T * ddp_world_size) == 0, f"make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = batch_size // (B * T * ddp_world_size)
    logger.info(f"total desired batch size: {batch_size}")
    logger.info(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # data loader related to its process_rank
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader   = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

    # create model
    vocab_size = 50304
    model = GPT(GPTconfig(vocab_size=vocab_size))
    model.to(device)
    raw_model = model

    # lr scheduler & max steps
    max_lr = 6e-4
    min_lr = 0.1 * max_lr   # 10% of max_lr
    warmup_steps   = 715    # first 350M tokens
    max_steps      = 19073
    val_loss_steps = 20

    def get_lr(it):
        if it < warmup_steps:
            return (max_lr / warmup_steps) * (it + 1)
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    weight_decay = 0.1
    optimizer = model.configure_optimizer(weight_decay=weight_decay, lr=max_lr, device=device)

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # compile the model
    logger.info("compiling the model...")
    model = torch.compile(model)

    # logger config
    logger.set_config({
        **vars(GPTconfig(vocab_size=vocab_size)),
        "batch_size": batch_size, "B": B, "T": T,
        "max_lr": max_lr, "min_lr": min_lr,
        "warmup_steps": warmup_steps, "max_steps": max_steps,
        "world_size": ddp_world_size, "weight_decay": weight_decay,
    })

    # main loop
    for step in range(max_steps):

        # validation
        if step % 100 == 0:

            model.eval()

            if device.startswith("cuda"): torch.cuda.synchronize()
            t0 = time.time()

            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = torch.zeros((), device=device)
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss /= val_loss_steps
                    val_loss_accum += loss.detach()

            # average val_loss_accum across all processes
            if ddp:
                torch.distributed.all_reduce(val_loss_accum, op=torch.distributed.ReduceOp.AVG)

            if device.startswith("cuda"): torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0

            logger.metric(step=step, split="val", loss=val_loss_accum.item(), dt=dt)
                

        # sample tokens from the model
        if step > 0 and step % 1000 == 0:
            raw_model.eval()
            num_return_sequences = 4
            max_length = 32
            enc = tiktoken.get_encoding('gpt2')
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (num_return_sequences, tokens.shape)
            xgen = tokens.to(device) 
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        # use raw_model to avoid recompilation
                        logits, loss = raw_model(xgen) # (B, T, vocab_size)
                    logits = logits[:, -1, :]  # (B, vocab_size)
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50
                    # topk_probs is (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                    xgen = torch.cat((xgen, xcol), dim=1)

            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                logger.info(f"rank {ddp_rank} sample {i} => {decoded}")

        # train
        model.train()

        if device.startswith("cuda"): torch.cuda.synchronize()
        t0 = time.time()

        optimizer.zero_grad()
        loss_accum = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss /= grad_accum_steps
            loss_accum += loss.detach()
            # sync gradients only after the last micro batch
            if ddp: 
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward()

        # average loss_accum across all processes
        if ddp: 
            torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)
        
        # gradients cliping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # sync with lr schedule
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        if device.startswith("cuda"): torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt

        # logging
        logger.metric(step=step, split="train", loss=loss_accum.item(),
                      lr=lr, grad_norm=norm.item(), dt=dt, tok_per_sec=tokens_per_sec,
                      mem_gb=torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0)

        # save checkpoint
        if master_process and (step > 0 and (step % 5000 == 0 or step == max_steps - 1)):
            logger.info(f"saving checkpoint at step {step}...")
            torch.save({
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': raw_model.config,
                'train_loader': {'shard': train_loader.current_shard, 
                                 'position': train_loader.current_position},
                'rng': {'torch': torch.get_rng_state(), 
                        'cuda': torch.cuda.get_rng_state_all() if device.startswith("cuda") else None},
                'val_loss': val_loss_accum.item(),
                'run_name': run_name,
            }, f"{log_dir}/{run_name}/ckpt_{step:05d}.pt")
            
    # close logger
    logger.close()

    # cleanup all processes
    if ddp: destroy_process_group()
