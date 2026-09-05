"""
HellaSwag validation for the nanoGPT-style model in gpt.py

HellaSwag: https://github.com/rowanz/hellaswag
Each example is a context plus 4 candidate endings, exactly one of which is correct.

Evaluation protocol (same as the LM-harness "acc_norm"):
  for each of the 4 candidates, build [context tokens] + [ending tokens],
  compute the average cross-entropy over the ENDING tokens only,
  and pick the candidate with the lowest average loss.

Averaging (rather than summing) normalises for ending length, otherwise the
model just prefers whichever ending happens to be shortest.

Standalone use:
    python hellaswag.py --ckpt log/ckpt_19072.pt
    python hellaswag.py --hf gpt2              # HuggingFace baseline
    torchrun --nproc_per_node=8 hellaswag.py --ckpt log/ckpt_19072.pt

Reference numbers (acc_norm on the 10042-example val split):
    random                 25.0%
    GPT-2  124M            29.6%
    GPT-3  124M            33.7%
"""

import os
import sys
import json
import argparse
import urllib.request

import torch
import tiktoken
import torch.distributed as dist
from torch.nn import functional as F
from tqdm import tqdm

import gpt
from gpt import GPT, GPTconfig

# checkpoints saved before the gpt2.py -> gpt.py split pickled the config as gpt2.GPTconfig
sys.modules.setdefault("gpt2", gpt)

DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hellaswag")
DATA_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"

enc = tiktoken.get_encoding("gpt2")


# ---------------------------

def download(split="val"):
    """Downloads the HellaSwag split to DATA_CACHE_DIR if it isn't there yet."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    path = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(path):
        print(f"downloading {DATA_URL} -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def iterate_examples(split="val"):
    path = download(split)
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def render_example(example, block_size=1024):
    """
    Returns (tokens, mask, label):
        tokens [4, N] int64, right-padded with zeros
        mask   [4, N] int64, 1 on ending tokens, 0 on context and padding
        label  int, index of the correct ending
    Returns None if the example does not fit into block_size.
    """
    ctx_tokens = enc.encode(example["ctx"])

    tok_rows, mask_rows = [], []
    for ending in example["endings"]:
        # the leading space matters: GPT-2 BPE encodes " word" and "word" differently
        end_tokens = enc.encode(" " + ending)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    max_len = max(len(row) for row in tok_rows)
    if max_len > block_size:
        return None

    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (t_row, m_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(t_row)] = torch.tensor(t_row, dtype=torch.long)
        mask[i, : len(m_row)] = torch.tensor(m_row, dtype=torch.long)

    return tokens, mask, int(example["label"])


# ---------------------------

def get_most_likely_row(tokens, mask, logits):
    """
    Given logits [4, N, V] for the 4 candidates, return (pred_sum, pred_avg):
    the argmin of the summed and of the length-normalised ending loss.
    """
    # next-token alignment: logits at position i predict token i+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()

    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_tokens.view(-1),
        reduction="none",
    ).view(tokens.size(0), -1)

    shift_mask = mask[..., 1:].contiguous()
    masked = losses * shift_mask

    sum_loss = masked.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1).clamp(min=1)

    return sum_loss.argmin().item(), avg_loss.argmin().item()


# ---------------------------

@torch.no_grad()
def evaluate_hellaswag(model, device, ddp_rank=0, ddp_world_size=1, limit=None, progress=False):
    """
    Evaluates `model` on the HellaSwag val split.

    IMPORTANT: pass the *uncompiled* module (raw_model). Every example has a
    different sequence length, so a torch.compile'd model would recompile on
    essentially every step.

    Under DDP each rank takes every ddp_world_size-th example; counts are
    all-reduced at the end, so every rank returns the same numbers.
    """
    was_training = model.training
    model.eval()

    device_type = "cuda" if str(device).startswith("cuda") else str(device)
    block_size = model.config.block_size

    n_total = n_correct = n_correct_norm = n_skipped = 0

    stream = enumerate(iterate_examples("val"))
    if progress and ddp_rank == 0:
        stream = tqdm(stream, desc="hellaswag", unit="ex")

    for i, example in stream:
        if limit is not None and i >= limit:
            break
        if i % ddp_world_size != ddp_rank:
            continue

        rendered = render_example(example, block_size=block_size)
        if rendered is None:
            n_skipped += 1
            continue
        tokens, mask, label = rendered
        tokens, mask = tokens.to(device), mask.to(device)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(tokens)

        pred, pred_norm = get_most_likely_row(tokens, mask, logits.float())

        n_total += 1
        n_correct += int(pred == label)
        n_correct_norm += int(pred_norm == label)

    stats = torch.tensor(
        [n_total, n_correct, n_correct_norm, n_skipped], dtype=torch.long, device=device
    )
    if ddp_world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    n_total, n_correct, n_correct_norm, n_skipped = stats.tolist()

    if was_training:
        model.train()

    return {
        "num_total": n_total,
        "num_skipped": n_skipped,
        "acc": n_correct / max(n_total, 1),
        "acc_norm": n_correct_norm / max(n_total, 1),
    }


# ---------------------------

def load_model_from_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt.get("config") or GPTconfig()
    if isinstance(config, dict):
        config = GPTconfig(**config)
    model = GPT(config)

    state = ckpt["model"]
    # strip wrapper prefixes in case the checkpoint was saved from a wrapped module
    for prefix in ("_orig_mod.", "module."):
        if any(k.startswith(prefix) for k in state):
            state = {k.removeprefix(prefix): v for k, v in state.items()}

    model.load_state_dict(state)
    print(f"loaded checkpoint {path} (step {ckpt.get('step', '?')})")
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="path to a .pt checkpoint")
    parser.add_argument("--hf", type=str, default=None, help="HuggingFace model, e.g. gpt2")
    parser.add_argument("--limit", type=int, default=None, help="only the first N examples")
    args = parser.parse_args()
    assert bool(args.ckpt) ^ bool(args.hf), "pass exactly one of --ckpt / --hf"

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
    else:
        ddp_rank, ddp_world_size = 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.set_float32_matmul_precision("high")

    if args.ckpt:
        model = load_model_from_checkpoint(args.ckpt, device)
    else:
        model = GPT.from_pretrained(args.hf).to(device)

    result = evaluate_hellaswag(
        model, device, ddp_rank, ddp_world_size, limit=args.limit, progress=True
    )

    if ddp_rank == 0:
        print(
            f"hellaswag: {result['num_total']} examples | "
            f"acc {result['acc']:.4f} | acc_norm {result['acc_norm']:.4f}"
            + (f" | skipped {result['num_skipped']}" if result["num_skipped"] else "")
        )

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()