"""Tokenize C4 (en) with the GPT-2 BPE into flat uint16 files: data/c4/{train,val}.bin

    python prepare_data.py                       # 600M train tokens + 5M val tokens
    python prepare_data.py --train_tokens 300e6  # smaller / faster
    python prepare_data.py --synthetic           # tiny learnable fake data (smoke tests only)

Validation comes from the official C4 validation split, so it never overlaps train.
"""
import argparse
import json
import os

import numpy as np


def synthetic(out, n_train, n_val):
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", n_val)):
        x = np.empty(n, dtype=np.uint16)
        x[0] = 1
        noise = rng.random(n) < 0.3
        rnd = rng.integers(0, 512, n)
        for i in range(1, n):  # next = f(prev) with prob 0.7, so loss can fall well below ln(512)
            x[i] = rnd[i] if noise[i] else (3 * int(x[i - 1]) + 7) % 512
        x.tofile(os.path.join(out, f"{split}.bin"))


def real(out, n_train, n_val, dataset, config):
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    for split, hf_split, n in (("val", "validation", n_val), ("train", "train", n_train)):
        ds = load_dataset(dataset, config, split=hf_split, streaming=True)
        path = os.path.join(out, f"{split}.bin")
        written, batch = 0, []
        with open(path, "wb") as f:
            def flush():
                nonlocal written, batch
                for ids in enc.encode_ordinary_batch(batch, num_threads=8):
                    arr = np.array(ids + [eot], dtype=np.uint16)
                    arr = arr[: n - written]
                    arr.tofile(f)
                    written += len(arr)
                    if written >= n:
                        break
                batch = []

            for ex in ds:
                batch.append(ex["text"])
                if len(batch) == 1024:
                    flush()
                    print(f"{split}: {written/1e6:.1f}M / {n/1e6:.0f}M tokens", flush=True)
                    if written >= n:
                        break
            if written < n and batch:
                flush()
        print(f"{split}: wrote {written} tokens -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/c4")
    ap.add_argument("--train_tokens", type=float, default=600e6)
    ap.add_argument("--val_tokens", type=float, default=5e6)
    ap.add_argument("--dataset", default="allenai/c4")
    ap.add_argument("--config", default="en")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nt, nv = int(a.train_tokens), int(a.val_tokens)
    if a.synthetic:
        synthetic(a.out, nt, nv)
    else:
        real(a.out, nt, nv, a.dataset, a.config)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"train_tokens": nt, "val_tokens": nv, "synthetic": a.synthetic,
                   "dataset": None if a.synthetic else f"{a.dataset}/{a.config}", "tokenizer": "gpt2-bpe"}, f)
