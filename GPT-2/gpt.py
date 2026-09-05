import inspect
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F

# -----------------------------------


@dataclass
class GPTconfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_emb: int = 768


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

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def configure_optimizer(self, weight_decay, lr, device):
        """Builds AdamW with weight decay applied only to >=2D tensors (matmuls, embeddings).
        Logging of the parameter counts is left to the caller."""
        params_dict = {p_name : p for p_name, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for p in params_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in params_dict.values() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in str(device)
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
