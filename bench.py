"""Measure training throughput (tokens/s) per model size on THIS machine -> bench.json.
grid.py uses it to estimate wall-clock time and to fit the grid into your time budget.

    python bench.py                 # all sizes, ~1 min each at most
    python bench.py --micro_bsz 16  # if you hit CUDA out-of-memory
"""
import argparse
import json
import time

import torch

from scaling.model import GPT, GPTConfig, flops_per_token
from scaling.sizes import SEQ_LEN, SIZES

ap = argparse.ArgumentParser()
ap.add_argument("--micro_bsz", type=int, default=32)
ap.add_argument("--steps", type=int, default=15)
ap.add_argument("--compile", action="store_true")
a = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
amp = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda"))
out = {"_gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "_micro_bsz": a.micro_bsz}
for name, s in SIZES.items():
    cfg = GPTConfig(seq_len=SEQ_LEN, **s)
    model = GPT(cfg).to(device)
    fwd = torch.compile(model) if a.compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, 50000, (a.micro_bsz, SEQ_LEN), device=device)
    for i in range(a.steps + 3):
        if i == 3:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        with amp:
            _, loss = fwd(x, x)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.synchronize()
    tps = a.steps * a.micro_bsz * SEQ_LEN / (time.time() - t0)
    out[name] = tps
    print(f"{name}: {tps:,.0f} tok/s  ({tps * flops_per_token(cfg) / 1e12:.1f} TFLOPs achieved)", flush=True)
    del model, opt, fwd
    if device == "cuda":
        torch.cuda.empty_cache()
json.dump(out, open("bench.json", "w"), indent=1)
print("wrote bench.json")
