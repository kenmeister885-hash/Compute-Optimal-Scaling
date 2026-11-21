"""Write fake runs/*.json from a KNOWN scaling law so fit.py can be checked end to end.

Truth: L = E + A/N^alpha + B/D^beta with N = the "total" parameter count,
       (E, A, B, alpha, beta) = (1.69, 700, 410.7, 0.34, 0.28)  [Hoffmann-like; A raised so the optimum lies inside our ladder]
plus Gaussian noise. Because we choose the law, the TRUE compute-optimal N*, D* are computable with
fit.param_opt (see tests/test_fit_recovery.py), so we know what fit.py should recover.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaling.sizes import BATCH_TOKENS, SEQ_LEN, SIZES, VOCAB_SIZE, flops_per_token, nonembed_params  # noqa: E402
from grid import plan  # noqa: E402

E, A, B, AL, BE = 1.69, 700.0, 410.7, 0.34, 0.28


def make(out_dir, budgets, noise=0.005, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    for j in plan(budgets, 600e6, 0.3, 200, 100, {}):
        s = SIZES[j["size"]]
        L, d = s["n_layer"], s["d_model"]
        body = nonembed_params(L, d)
        emb_in, head = VOCAB_SIZE * d + SEQ_LEN * d, VOCAB_SIZE * d
        params = dict(emb_in=emb_in, head=head, body=body, total=emb_in + head + body,
                      nonembed=body, nonembed_plus_head=body + head)
        N, D = params["total"], j["tokens"]
        loss = E + A / N**AL + B / D**BE + rng.normal(0, noise)
        # idealised training curve: loss after t steps follows the same law at D_t = t * batch_tokens
        curve = [dict(step=t, train_loss=None, val_loss=E + A / N**AL + B / (t * BATCH_TOKENS) ** BE + rng.normal(0, noise))
                 for t in sorted({max(1, int(j["steps"] * f)) for f in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)})]
        fpt = flops_per_token(L, d, SEQ_LEN)
        name = f"{j['size']}_C{j['budget']:.0e}"
        json.dump(dict(name=name, size=j["size"], **s, seq_len=SEQ_LEN, steps=j["steps"], batch_tokens=BATCH_TOKENS,
                       tokens=D, flops=D * fpt, flops_per_token=fpt, budget=j["budget"], params=params,
                       val_loss=loss, curve=curve, wall_seconds=1.0, gpu="synthetic"),
                  open(os.path.join(out_dir, name + ".json"), "w"))


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "runs_synthetic", [3e15, 1e16, 3e16, 1e17])
