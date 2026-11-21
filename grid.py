"""Plan/runthe IsoFLOP training grid.

    python grid.py                          # print the plan with time estimates
    python grid.py --run                    # train everything, smallest budgets first (resumable)
    python grid.py --max_hours 3 --run      # drop the most expensive budgets until it fits
    python grid.py --shard 0/4 --run        # cluster: this process runs every 4th job
"""
import argparse
import json
import os
import subprocess
import sys

from scaling.sizes import BATCH_TOKENS, SEQ_LEN, SIZES, flops_per_token, nonembed_params

PRESETS = {
    "fast": [3e15, 1e16, 3e16],  # ~2e17 FLOPs total, 12 runs; needs >=310M prepared train tokens
    "full": [3e15, 1e16, 3e16, 1e17],  # adds ~3e17 FLOPs (3 runs); needs >=480M tokens
}


def plan(budgets, max_tokens, min_ratio, max_ratio, min_steps, bench):
    jobs = []
    for C in budgets:
        for size, s in SIZES.items():
            fpt = flops_per_token(s["n_layer"], s["d_model"], SEQ_LEN)
            steps = round(C / fpt / BATCH_TOKENS)
            D = steps * BATCH_TOKENS
            n_ne = nonembed_params(s["n_layer"], s["d_model"])
            ratio = D / n_ne
            if steps < min_steps or D > max_tokens or not (min_ratio <= ratio <= max_ratio):
                continue
            tps = bench.get(size)
            jobs.append(dict(size=size, budget=C, tokens=D, steps=steps, flops=D * fpt,
                             tokens_per_nonembed_param=round(ratio, 2),
                             est_hours=(D / tps / 3600) if tps else None))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fast", choices=list(PRESETS))
    ap.add_argument("--budgets", type=float, nargs="*", default=None)
    ap.add_argument("--max_tokens", type=float, default=None,
                    help="size of prepared train data (default: read from data/c4/meta.json)")
    ap.add_argument("--min_ratio", type=float, default=0.5, help="min tokens per non-embedding param")
    ap.add_argument("--max_ratio", type=float, default=200)
    ap.add_argument("--min_steps", type=int, default=150)
    ap.add_argument("--max_hours", type=float, default=None)
    ap.add_argument("--bench", default="bench.json")
    ap.add_argument("--data_dir", default="data/c4")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--micro_bsz", type=int, default=None)
    ap.add_argument("--shard", default=None, help="k/n")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    if a.max_tokens is None:
        meta = os.path.join(a.data_dir, "meta.json")
        a.max_tokens = json.load(open(meta))["train_tokens"] if os.path.exists(meta) else 600e6
    print(f"available train tokens: {a.max_tokens/1e6:.0f}M")
    bench = json.load(open(a.bench)) if os.path.exists(a.bench) else {}
    if not bench:
        print("(no bench.json -> run `python bench.py` first for time estimates)\n")
    budgets = sorted(a.budgets or PRESETS[a.preset])
    jobs = plan(budgets, a.max_tokens, a.min_ratio, a.max_ratio, a.min_steps, bench)

    if a.max_hours and bench:
        while jobs and sum(j["est_hours"] for j in jobs) > a.max_hours:
            top = max(j["budget"] for j in jobs)
            jobs = [j for j in jobs if j["budget"] != top]
            print(f"dropping budget {top:.0e} to fit --max_hours {a.max_hours}")

    print(f"{'size':<5}{'budget':>9}{'tokens(M)':>11}{'steps':>7}{'tok/nonemb':>11}{'est_h':>8}")
    for j in jobs:
        eh = f"{j['est_hours']:.2f}" if j["est_hours"] is not None else "?"
        print(f"{j['size']:<5}{j['budget']:>9.0e}{j['tokens']/1e6:>11.1f}{j['steps']:>7}{j['tokens_per_nonembed_param']:>11}{eh:>8}")
    per_budget = {b: sum(1 for j in jobs if j["budget"] == b) for b in budgets}
    print("\nruns per budget:", {f"{b:.0e}": n for b, n in per_budget.items()}, "(need >=3 per budget for a parabola)")
    if bench:
        print(f"estimated total: {sum(j['est_hours'] for j in jobs):.1f} GPU-hours  ({len(jobs)} runs)")
    json.dump(jobs, open("grid.json", "w"), indent=1)

    if not a.run:
        return
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        jobs = jobs[k::n]
    for j in jobs:
        name = f"{j['size']}_C{j['budget']:.0e}".replace("+", "")
        cmd = [sys.executable, "train.py", "--size", j["size"], "--tokens", str(j["tokens"]),
               "--budget", str(j["budget"]), "--name", name, "--data_dir", a.data_dir, "--out_dir", a.out_dir]
        if a.micro_bsz:
            cmd += ["--micro_bsz", str(a.micro_bsz)]
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
