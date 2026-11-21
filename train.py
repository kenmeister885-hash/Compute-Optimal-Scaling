"""Train ONE model for a fixed token budget and write runs/<name>.json.

Each run has its own warmup + cosine schedule (decayed to 10% of peak LR over its own
length), which is what IsoFLOP-style fits assume. Data is consumed in order, once
(no repeated tokens).

Example:
    python train.py --size s1 --tokens 100000000 --budget 1e16
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from scaling.model import GPT, GPTConfig, count_params, flops_per_token
from scaling.sizes import BATCH_TOKENS, SEQ_LEN, SIZES


def default_lr(d_model: int) -> float:
    # Untuned heuristic (~6e-4 at d=768, scaling ~1/d, capped). Documented as a limitation.
    return min(2e-3, 6e-4 * 768 / d_model)


def get_lr(step, steps, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=list(SIZES))
    ap.add_argument("--tokens", type=float, required=True, help="training tokens D")
    ap.add_argument("--budget", type=float, default=0.0, help="nominal FLOP budget label (for grouping)")
    ap.add_argument("--data_dir", default="data/c4")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--micro_bsz", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_batches", type=int, default=40, help="batches for the FINAL validation loss")
    ap.add_argument("--eval_batches_mid", type=int, default=10, help="batches for intermediate evals (Approach 1 curves)")
    ap.add_argument("--eval_every", type=int, default=None, help="default: ~8 evals per run (min every 50 steps)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    steps = max(1, round(a.tokens / BATCH_TOKENS))
    D = steps * BATCH_TOKENS
    name = a.name or f"{a.size}_D{int(D / 1e6)}M"
    os.makedirs(a.out_dir, exist_ok=True)
    out_json = os.path.join(a.out_dir, name + ".json")
    if os.path.exists(out_json) and not a.force:
        print(f"[skip] {out_json} exists")
        return

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GPTConfig(seq_len=SEQ_LEN, **SIZES[a.size])
    model = GPT(cfg).to(device)
    params = count_params(model)
    fpt = flops_per_token(cfg)
    train = np.memmap(os.path.join(a.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(a.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    assert len(train) >= D + 1, f"need {D} train tokens, have {len(train)} (no repetition allowed)"

    T = SEQ_LEN
    micro_tokens = a.micro_bsz * T
    assert BATCH_TOKENS % micro_tokens == 0
    accum = BATCH_TOKENS // micro_tokens
    peak = a.lr or default_lr(cfg.d_model)
    warmup = max(10, min(200, steps // 20))

    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=peak, betas=(0.9, 0.95), fused=(device == "cuda"),
    )
    fwd = torch.compile(model) if a.compile else model
    amp = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda"))

    def batch(data, start):
        blk = torch.from_numpy(data[start : start + micro_tokens + 1].astype(np.int64))
        return blk[:-1].view(a.micro_bsz, T).to(device), blk[1:].view(a.micro_bsz, T).to(device)

    @torch.no_grad()
    def evaluate(n_batches):
        model.eval()
        tot = 0.0
        for i in range(n_batches):
            x, y = batch(val, i * micro_tokens)
            with amp:
                _, loss = fwd(x, y)
            tot += loss.item()
        model.train()
        return tot / n_batches

    eval_every = a.eval_every or max(50, steps // 8)

    print(f"{name}: size={a.size} N={params} steps={steps} D={D/1e6:.1f}M flops={D*fpt:.3e} lr={peak:.2e}")
    curve = []
    t0 = time.time()
    run_loss = 0.0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = get_lr(step, steps, peak, warmup)
        for m in range(accum):
            x, y = batch(train, (step * accum + m) * micro_tokens)
            with amp:
                _, loss = fwd(x, y)
            (loss / accum).backward()
            run_loss += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % eval_every == 0 and step + 1 < steps:
            ev = evaluate(a.eval_batches_mid)  # intermediate point on the loss-vs-FLOPs curve (Approach 1)
            tl = run_loss / eval_every
            curve.append({"step": step + 1, "train_loss": tl, "val_loss": ev})
            print(f"  step {step+1}/{steps} train {tl:.4f} val {ev:.4f} ({time.time()-t0:.0f}s)", flush=True)
            run_loss = 0.0
    wall = time.time() - t0
    val_loss = evaluate(a.eval_batches)
    curve.append({"step": steps, "train_loss": None, "val_loss": val_loss})
    res = {
        "name": name, "size": a.size, **SIZES[a.size], "seq_len": T,
        "steps": steps, "batch_tokens": BATCH_TOKENS, "tokens": D,
        "flops": D * fpt, "flops_per_token": fpt, "budget": a.budget or D * fpt,
        "params": params, "lr": peak, "seed": a.seed,
        "val_loss": val_loss, "wall_seconds": wall, "tokens_per_sec": D / wall,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__, "curve": curve,
    }
    with open(out_json, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[done] {name} val_loss={val_loss:.4f} wall={wall/60:.1f} min")


if __name__ == "__main__":
    main()
