import gpt2
import time
import torch 
import tiktoken
from torch.nn import functional as F
from gpt2 import GPT, GPTconfig

# ----------------------

class KVcache():

    def __init__(self, shape, device, dtype=torch.bfloat16):
        self.k = torch.empty(shape, device=device, dtype=dtype) # (B, n_head, 0, head_size)
        self.v = torch.empty(shape, device=device, dtype=dtype) # (B, n_head, 0, head_size)

    def update(self, k, v):
        self.k = torch.cat([self.k, k], dim=2)
        self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def __len__(self):
        return self.k.size(2)


class GPT(GPT):

    def __init__(self, config):
        super().__init__(config)
        self.caches = None

    def forward(self, idx, targets=None, use_cache=False):
        B, T = idx.size()
        assert T <= self.config.block_size, f"model's input can't be more than {self.config.block_size} tokens"

        if use_cache:
            if self.caches is None:
                shape = (B, self.config.n_head, 0, self.config.n_emb // self.config.n_head)
                self.caches = [KVcache(shape, idx.device) for _ in range(self.config.n_layer)]
            kv_cache_len = len(self.caches[0])
        else:
            kv_cache_len = 0

        pos = torch.arange(kv_cache_len, T + kv_cache_len, 
                            dtype=torch.long, device=idx.device) # (T)
        pos_emb = self.transformer.wpe(pos) # (T, n_emb)
        tok_emb = self.transformer.wte(idx) # (B, T, n_emb)

        x = tok_emb + pos_emb # (B, T, n_emb)
        for i, block in enumerate(self.transformer.h):
            if use_cache:
                x = block(x, self.caches[i])
            else:
                x = block(x)

        x = self.transformer.ln_f(x) 
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, logits.size(-1)), targets.view(B * T))

        return logits, loss


def attn_forward(self, x, cache=None):
    B, T, C = x.shape
    qkv = self.c_attn(x) # (B, T, 3 * C)
    q, k, v = qkv.split(self.n_emb, dim=2) # (B, T, C) 3 times
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    if cache is not None:
        k, v = cache.update(k, v)

    y = F.scaled_dot_product_attention(q, k, v, is_causal=(k.size(2) == T)) # flash attention

    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.c_proj(y)
    return y


def block_forward(self, x, cache=None):
    x = x + self.attn(self.ln_1(x), cache) # Pre Norm
    x = x + self.mlp(self.ln_2(x))
    return x


gpt2.CausalSelfAttention.forward = attn_forward
gpt2.Block.forward = block_forward


# ----------------------


if __name__ == "__main__":

    checkpoint_path = "log/ckpt_19072.pt"
    device = 'cuda'

    # reproducibility
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    # precision
    torch.set_float32_matmul_precision('high')

    print("loading model...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(ckpt['config'])
    sd = ckpt['model']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print(f"step {ckpt.get('step')}\n")

    # parameters of generation
    num_return_sequences = 32
    max_length = 64
    prompt = "Hello, I'm a language model,"
    print(f"parameters of generation: num_return_sequences = {num_return_sequences} | max_length = {max_length}")
    
    # encode
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (num_return_sequences, tokens.shape)
    xgen = tokens.to(device)
    xgen_copy = xgen.clone()

    # print the generated text
    def print_text(gen):
        for i in range(num_return_sequences):
            tokens = gen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"decoded sample {i} => {decoded}")
        print()

    # sampler
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)

    # warmup
    print("warmup...\n")
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits_warmup, _ = model(xgen)
    torch.cuda.synchronize()

    # ---- naive sampling -----

    t0 = time.time()

    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(xgen)                               # (B, T, vocab_size)
            logits = logits[:, -1, :]                                    # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)

            # top_k sampling
            topk_probs, topk_idx = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
            xcol = torch.gather(topk_idx, -1, ix)                        # (B, 1)

            # greedy sampling
            # xcol = torch.argmax(probs, dim=1).unsqueeze(-1)              # (B, 1)

            xgen = torch.cat((xgen, xcol), dim=1)
    
    torch.cuda.synchronize()
    t1 = time.time()
    dt_naive = t1 - t0
    print(f"naive sampling time: {dt_naive:.6f}s")

    print_text(xgen)

    # ---- kv cache ----

    sample_rng.manual_seed(42)

    t0 = time.time()

    # prefill & decode
    # first step will prefill all tokens and store them in cache
    # then model gets only 1 token using previous cached k, v
    cur = xgen_copy
    while xgen_copy.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(cur, use_cache=True)               # (B, T, vocab_size)
            logits = logits[:, -1, :]                                   # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)

            # top_k sampling
            topk_probs, topk_idx = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            cur = torch.gather(topk_idx, -1, ix)                        # (B, 1)

            # greedy sampling 
            # cur = torch.argmax(probs, dim=1).unsqueeze(-1)              # (B, 1)

            xgen_copy = torch.cat((xgen_copy, cur), dim=1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    print(f"kv cache sampling time: {dt:.6f}s")

    print_text(xgen_copy)

    print(f"speedup ratio: {(dt_naive / dt):.1f}")
