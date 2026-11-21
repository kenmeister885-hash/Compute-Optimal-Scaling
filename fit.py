"""Fit compute-optimal scaling laws from runs/*.json and test their robustness.

    python fit.py                       # tables + figures in results/
    python fit.py --boot 200            # + bootstrap 95% CIs (slower)
    python fit.py --skip_shrink         # skip the grid-shrinking sweep

What it does (maps to the research questions in the project doc):
  RQ1  Refit under several PARAMETER-COUNT CONVENTIONS (scaling/model.py:CONVENTIONS) with
       (a) IsoFLOP parabolas ("Approach 2") and (b) a parametric loss surface
       L = E + A/N^alpha + B/D^beta ("Approach 3"), and report how the compute-optimal
       exponents and the optimal token count D* change.
  RQ2  Re-run the same analysis on progressively smaller fitting grids (drop a budget,
       drop a model size, ...) to see when conclusions stop being stable.
  RQ3  Compare Approaches 1, 2 and 3 pairwise for every convention.

All three of Hoffmann et al.'s approaches are implemented: 1 = training-curve envelope,
2 = IsoFLOP parabolas, 3 = parametric loss surface. Everything is torch-free.
"""
import argparse
import csv
import glob
import json
import math
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

CONVENTIONS = ["total", "nonembed", "nonembed_plus_head"]  # keep in sync with scaling/model.py
HUBER_DELTA = 1e-3


# ----------------------------------------------------------------------------- data
def load_runs(run_dir):
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(run_dir, "*.json")))]
    return [r for r in runs if "val_loss" in r and "params" in r]


def ladder_fit(runs, conv):
    """log(flops/token) = p*log(N_conv) + k over the model ladder (used to convert N <-> D at fixed C)."""
    by = {r["size"]: (r["params"][conv], r["flops_per_token"]) for r in runs}
    if len(by) < 2:
        return None
    N = np.log([v[0] for v in by.values()])
    F = np.log([v[1] for v in by.values()])
    return tuple(np.polyfit(N, F, 1))


# ----------------------------------------------------------------------------- Approach 2
def approach2(runs, conv, c_ref):
    groups = {}
    for r in runs:
        groups.setdefault(r["budget"], []).append(r)
    pts = []
    for C, g in sorted(groups.items()):
        g = sorted(g, key=lambda r: r["params"][conv])
        if len(g) < 3:
            continue
        x = np.log([r["params"][conv] for r in g])
        y = np.array([r["val_loss"] for r in g])
        lD = np.log([r["tokens"] for r in g])
        if len(np.unique(np.round(x, 8))) < 3:  # bootstrap resamples can repeat points
            continue
        xm = x.mean()
        a, b, c = np.polyfit(x - xm, y, 2)  # centred for conditioning
        if a <= 0:
            continue
        xs = -b / (2 * a) + xm
        if not (x.min() <= xs <= x.max()):  # require an interior minimum
            continue
        pts.append((C, math.exp(xs), math.exp(np.interp(xs, x, lD)), c - b * b / (4 * a)))
    if len(pts) < 2:
        return None
    lc = np.log([p[0] for p in pts])
    na, nb = np.polyfit(lc, np.log([p[1] for p in pts]), 1)
    da, db = np.polyfit(lc, np.log([p[2] for p in pts]), 1)
    N_ref, D_ref = math.exp(nb + na * math.log(c_ref)), math.exp(db + da * math.log(c_ref))
    return dict(a=na, b=da, N_ref=N_ref, D_ref=D_ref, ratio_ref=D_ref / N_ref, n_budgets=len(pts), pts=pts,
                ratio_slope=_ratio_slope(na, nb, da, db, c_ref))


def _ratio_slope(na, nb, da, db, c_ref):
    """Change in the optimal tokens-per-parameter ratio per decade of compute, centred on c_ref."""
    f = lambda C: math.exp((db + da * math.log(C)) - (nb + na * math.log(C)))
    s = math.sqrt(10)
    return f(c_ref * s) - f(c_ref / s)


# ----------------------------------------------------------------------------- Approach 1
def curve_points(r):
    """(FLOPs, val_loss) points along one run's training curve (intermediate evals + final)."""
    pts = [(c["step"] * r["batch_tokens"] * r["flops_per_token"], c["val_loss"])
           for c in r.get("curve", []) if c.get("val_loss") is not None and c["step"] < r["steps"]]
    pts.append((r["flops"], r["val_loss"]))
    return np.array(sorted(set(pts)))


def approach1(runs, conv, c_ref, n_grid=300):
    """Training-curve envelope (Hoffmann et al.'s Approach 1): at each FLOP count take the run whose
    (linearly interpolated, in log C) loss is lowest; regress that run's N and D on C.
    Caveat: intermediate points come from runs whose cosine schedule has not finished decaying, so
    the envelope is dominated by final points; N* is a step function over the discrete size ladder."""
    curves = []
    for r in runs:
        p = curve_points(r)
        if len(p) >= 2:
            curves.append((r, np.log(p[:, 0]), p[:, 1]))
    if len(curves) < 2:
        return None
    grid = np.linspace(min(c[1].min() for c in curves), max(c[1].max() for c in curves), n_grid)
    rows = []
    for lc in grid:
        cand = [(np.interp(lc, x, y), r) for r, x, y in curves if x[0] <= lc <= x[-1]]
        if len({r["size"] for _, r in cand}) < 2:  # need at least two model sizes to compare
            continue
        _, r = min(cand, key=lambda t: t[0])
        rows.append((lc, r["params"][conv], math.exp(lc) / r["flops_per_token"]))
    if len(rows) < 10:
        return None
    lc = np.array([x[0] for x in rows])
    na, nb = np.polyfit(lc, np.log([x[1] for x in rows]), 1)
    da, db = np.polyfit(lc, np.log([x[2] for x in rows]), 1)
    N_ref, D_ref = math.exp(nb + na * math.log(c_ref)), math.exp(db + da * math.log(c_ref))
    return dict(a=na, b=da, N_ref=N_ref, D_ref=D_ref, ratio_ref=D_ref / N_ref, n_grid_pts=len(rows),
                ratio_slope=_ratio_slope(na, nb, da, db, c_ref))


# ----------------------------------------------------------------------------- Approach 3
def _huber(r, d):
    ar = np.abs(r)
    return np.where(ar <= d, 0.5 * r * r, d * (ar - 0.5 * d)), np.where(ar <= d, r, d * np.sign(r))


def fit_parametric(N, D, L, inits, delta=HUBER_DELTA):
    lN, lD, y = np.log(N), np.log(D), np.log(L)

    def f(th):
        la, lb, e, al, be = th
        z = np.stack([la - al * lN, lb - be * lD, np.full_like(lN, e)])
        pred = logsumexp(z, axis=0)
        w = np.exp(z - pred)
        val, g = _huber(pred - y, delta)
        grad = np.array([(g * w[0]).sum(), (g * w[1]).sum(), (g * w[2]).sum(),
                         (-g * w[0] * lN).sum(), (-g * w[1] * lD).sum()])
        return val.sum(), grad

    bounds = [(-30, 60), (-30, 60), (-10, math.log(L.min()) - 1e-6), (0.01, 3), (0.01, 3)]
    best = None
    for x0 in inits:
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best.fun


def default_inits(L):
    ub = math.log(L.min()) - 0.1
    return [[la, lb, min(e, ub), al, be] for la in (0, 8, 16) for lb in (0, 8, 16)
            for e in (-1, 0) for al in (0.1, 0.3, 0.6) for be in (0.1, 0.3, 0.6)]


def param_opt(theta, C, lad):
    """argmin_N of E + A N^-alpha + B D^-beta subject to D = C / flops_per_token(N)."""
    la, lb, e, al, be = theta
    p, k = lad
    lN = np.linspace(math.log(1e5), math.log(1e10), 4000)
    lD = math.log(C) - (p * lN + k)
    loss = np.exp(e) + np.exp(la - al * lN) + np.exp(lb - be * lD)
    i = loss.argmin()
    return math.exp(lN[i]), math.exp(lD[i])


def approach3(runs, conv, c_ref, lad, inits=None):
    if len(runs) < 6 or lad is None:
        return None
    N = np.array([r["params"][conv] for r in runs], float)
    D = np.array([r["tokens"] for r in runs], float)
    L = np.array([r["val_loss"] for r in runs], float)
    theta, obj = fit_parametric(N, D, L, inits or default_inits(L))
    Nr, Dr = param_opt(theta, c_ref, lad)
    n_hi, _ = param_opt(theta, 2 * c_ref, lad)
    n_lo, _ = param_opt(theta, c_ref / 2, lad)
    a = (math.log(n_hi) - math.log(n_lo)) / (2 * math.log(2))
    sq = math.sqrt(10)
    (n_h, d_h), (n_l, d_l) = param_opt(theta, c_ref * sq, lad), param_opt(theta, c_ref / sq, lad)
    return dict(a=a, b=1 - a, N_ref=Nr, D_ref=Dr, ratio_ref=Dr / Nr, ratio_slope=d_h / n_h - d_l / n_l,
                alpha=theta[3], beta=theta[4],
                E=math.exp(theta[2]), A=math.exp(theta[0]), B=math.exp(theta[1]), obj=obj, theta=theta)


# ----------------------------------------------------------------------------- bootstrap
def bootstrap(runs, conv, c_ref, lad, B, rng, a3_theta=None):
    out = {"a1": [], "D1": [], "a2": [], "D2": [], "a3": [], "D3": []}
    groups = {}
    for r in runs:
        groups.setdefault(r["budget"], []).append(r)
    for _ in range(B):
        res = [g[i] for g in groups.values() for i in rng.integers(0, len(g), len(g))]
        r1 = approach1(res, conv, c_ref)
        if r1:
            out["a1"].append(r1["a"]); out["D1"].append(r1["D_ref"])
        r2 = approach2(res, conv, c_ref)
        if r2:
            out["a2"].append(r2["a"]); out["D2"].append(r2["D_ref"])
        if a3_theta is not None:
            res = [runs[i] for i in rng.integers(0, len(runs), len(runs))]
            inits = [a3_theta] + [a3_theta + rng.normal(0, 0.2, 5) for _ in range(6)]
            r3 = approach3(res, conv, c_ref, lad, inits)
            if r3:
                out["a3"].append(r3["a"]); out["D3"].append(r3["D_ref"])
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) >= 10 else None
    return {k: ci(v) for k, v in out.items()} | {"n_ok": {k: len(v) for k, v in out.items()}}


# ----------------------------------------------------------------------------- analysis
def analyse(runs, all_runs, c_ref, boot=0, rng=None):
    table = {}
    for conv in CONVENTIONS:
        lad = ladder_fit(all_runs, conv)
        r1 = approach1(runs, conv, c_ref)
        r2 = approach2(runs, conv, c_ref)
        r3 = approach3(runs, conv, c_ref, lad)
        row = {"a1": r1, "a2": r2, "a3": r3}
        if boot:
            row["ci"] = bootstrap(runs, conv, c_ref, lad, boot, rng, r3["theta"] if r3 else None)
        table[conv] = row
    return table


def spreads(table):
    """Across conventions: absolute spread of exponent a, relative spread of D*(C_ref)."""
    out = {}
    for ap in ("a1", "a2", "a3"):
        rows = [t[ap] for t in table.values() if t[ap]]
        if len(rows) < 2:
            out[ap] = None
            continue
        a = [r["a"] for r in rows]
        D = [r["D_ref"] for r in rows]
        out[ap] = dict(a_spread=max(a) - min(a), D_rel_spread=(max(D) - min(D)) / float(np.mean(D)))
    return out


def shrink_sweep(runs, c_ref):
    sizes = sorted({r["size"] for r in runs})
    budgets = sorted({r["budget"] for r in runs})
    subs = [("full", runs)]
    subs += [(f"drop_budget_{b:.0e}", [r for r in runs if r["budget"] != b]) for b in budgets]
    subs += [(f"drop_size_{s}", [r for r in runs if r["size"] != s]) for s in sizes]
    subs += [("no_extreme_sizes", [r for r in runs if r["size"] not in (sizes[0], sizes[-1])]),
             ("every_other_size", [r for r in runs if r["size"] in sizes[::2]])]
    rows = []
    for name, rs in subs:
        t = analyse(rs, runs, c_ref)
        sp = spreads(t)
        row = {"subset": name, "n_runs": len(rs)}
        for conv in CONVENTIONS:
            for ap in ("a1", "a2", "a3"):
                row[f"{ap}_{conv}"] = t[conv][ap]["a"] if t[conv][ap] else ""
                row[f"D_{ap}_{conv}_M"] = t[conv][ap]["D_ref"] / 1e6 if t[conv][ap] else ""
        for ap in ("a1", "a2", "a3"):
            row[f"{ap}_a_spread"] = sp[ap]["a_spread"] if sp[ap] else ""
            row[f"{ap}_D_rel_spread"] = sp[ap]["D_rel_spread"] if sp[ap] else ""
        rows.append(row)
    return rows


def compute_report(runs):
    return dict(n_runs=len(runs), total_flops=sum(r["flops"] for r in runs),
                total_gpu_hours=sum(r["wall_seconds"] for r in runs) / 3600,
                gpus=sorted({r.get("gpu", "?") for r in runs}),
                per_run_hours=dict(min=min(r["wall_seconds"] for r in runs) / 3600,
                                   max=max(r["wall_seconds"] for r in runs) / 3600))


# ----------------------------------------------------------------------------- structured perturbations of N
# Following the four structured perturbations of the parameter counts in Schaeffer et al. (2025):
# multiplicative constant, additive constant, systematic (size-dependent) bias, log-normal noise.
# Applied to the base convention's N; D and the loss values are left untouched.
PERTURB_BASE = "total"
N_NOISE_DRAWS = 10
PERTURBATIONS = {
    "mult": ("multiplicative constant $c$", [0.1, 0.3, 1.0, 3.0, 10.0]),
    "add": ("additive offset $c$ (M params)", [-1e7, 0.0, 1e7, 4e7]),
    "bias": ("systematic bias slope $s$", [0.5, 0.75, 1.0, 1.5, 2.0]),
    "noise": ("log-normal noise $\\sigma$", [0.01, 0.1, 0.3]),
}


def perturb_runs(runs, kind, level, rng, base=PERTURB_BASE):
    """Copies of `runs` with params['pert'] = perturbed N. Perturbation is per model SIZE (a counting
    error is a property of the architecture, so every run of one size shares it)."""
    Nb = {}
    for r in runs:
        Nb.setdefault(r["size"], float(r["params"][base]))
    if kind == "mult":
        Np = {k: v * level for k, v in Nb.items()}
    elif kind == "add":
        Np = {k: v + level for k, v in Nb.items()}
    elif kind == "bias":  # stretch log N around the geometric mean with slope `level`
        g = math.exp(np.mean(np.log(list(Nb.values()))))
        Np = {k: g * (v / g) ** level for k, v in Nb.items()}
    elif kind == "noise":
        Np = {k: v * math.exp(rng.normal(0, level)) for k, v in Nb.items()}
    else:
        raise ValueError(kind)
    assert min(Np.values()) > 0, "perturbation made a parameter count non-positive"
    return [dict(r, params={**r["params"], "pert": Np[r["size"]]}) for r in runs]


def analyse_conv(runs, conv, c_ref):
    lad = ladder_fit(runs, conv)
    return {"a1": approach1(runs, conv, c_ref), "a2": approach2(runs, conv, c_ref),
            "a3": approach3(runs, conv, c_ref, lad)}


KEYS = ("a", "D_ref", "N_ref", "ratio_ref", "ratio_slope")


def perturbation_sweep(runs, c_ref, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for kind, (_, levels) in PERTURBATIONS.items():
        for lv in levels:
            for d in range(N_NOISE_DRAWS if kind == "noise" else 1):
                res = analyse_conv(perturb_runs(runs, kind, lv, rng), "pert", c_ref)
                row = {"kind": kind, "level": lv, "draw": d}
                for ap in ("a1", "a2", "a3"):
                    for k in KEYS:
                        row[f"{ap}_{k}"] = res[ap][k] if res[ap] else ""
                rows.append(row)
    return rows


def fig_perturbation(rows, path):
    import matplotlib.pyplot as plt
    kinds = list(PERTURBATIONS)
    fig, axes = plt.subplots(2, len(kinds), figsize=(10, 4.4))
    spec = (("a1", "Approach 1", AQUA, "D"), ("a2", "Approach 2", BLUE, "o"), ("a3", "Approach 3", ORANGE, "s"))
    for j, kind in enumerate(kinds):
        for i, (key, ylab) in enumerate((("ratio_ref", "tokens / param at $C_{ref}$"), ("a", "exponent $a$"))):
            ax = axes[i, j]
            for ap, label, col, mk in spec:
                by = {}
                for r in rows:
                    if r["kind"] == kind and r[f"{ap}_{key}"] != "":
                        by.setdefault(r["level"], []).append(r[f"{ap}_{key}"])
                xs = sorted(by)
                if not xs:
                    continue
                ax.errorbar([x / 1e6 if kind == "add" else x for x in xs], [np.mean(by[x]) for x in xs],
                            yerr=[np.std(by[x]) for x in xs], fmt=mk + "-", color=col, ms=5, lw=1.2, capsize=2,
                            label=label if (i == 0 and j == 0) else None)
            if kind in ("mult", "noise"):
                ax.set_xscale("log")
            if key == "ratio_ref":
                ax.set_yscale("log")
            ax.set_xlabel(PERTURBATIONS[kind][0], fontsize=8)
            if j == 0:
                ax.set_ylabel(ylab)
            _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=7, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(path + ".png", dpi=200); fig.savefig(path + ".pdf"); plt.close(fig)


# ----------------------------------------------------------------------------- output
def write_latex(table, path):
    names = {"total": "total", "nonembed": "non-embedding", "nonembed_plus_head": "excl.\\ input emb."}
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Fit & $N$ convention & exponent $a$ & $D^*$ (M tok) & $N^*$ (M) & $D^*/N^*$ \\", r"\midrule"]
    for ap, label in (("a1", "Envelope"), ("a2", "IsoFLOP"), ("a3", "Parametric")):
        for conv in CONVENTIONS:
            r = table[conv][ap]
            if r:
                lines.append(f"{label} & {names[conv]} & {r['a']:.3f} & {r['D_ref']/1e6:.0f} & "
                             f"{r['N_ref']/1e6:.1f} & {r['ratio_ref']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUES = ["#9ec5f2", "#5b9ae0", "#2a78d6", "#174a8c"]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)


def fig_isoflop(runs, conv, a2, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    budgets = sorted({r["budget"] for r in runs})
    for i, C in enumerate(budgets):
        col = BLUES[min(i + (4 - len(budgets)), 3)]
        g = sorted([r for r in runs if r["budget"] == C], key=lambda r: r["params"][conv])
        x = np.array([r["params"][conv] for r in g], float)
        y = np.array([r["val_loss"] for r in g])
        ax.plot(x, y, "o", color=col, ms=5, mec="white", mew=0.8)
        if len(g) >= 3:
            c2, c1, c0 = np.polyfit(np.log(x), y, 2)
            xx = np.linspace(np.log(x.min()), np.log(x.max()), 100)
            ax.plot(np.exp(xx), c2 * xx**2 + c1 * xx + c0, "-", color=col, lw=1.6)
        ax.annotate(f"{C:.0e}".replace("e+", "e"), (x[-1], y[-1]), xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
    if a2:
        ax.plot([p[1] for p in a2["pts"]], [p[3] for p in a2["pts"]], "k--", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(f"parameters N ({conv})"); ax.set_ylabel("validation loss (nats)")
    ax.set_title("IsoFLOP curves (label = FLOP budget; dashed = fitted optima)", fontsize=9, color=INK, loc="left")
    _style(ax)
    fig.tight_layout(); fig.savefig(path + ".png", dpi=200); fig.savefig(path + ".pdf"); plt.close(fig)


def fig_exponents(table, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    spec = (("a1", "Training-curve envelope (Approach 1)", AQUA, "D", -0.2),
            ("a2", "IsoFLOP parabola fit (Approach 2)", BLUE, "o", 0.0),
            ("a3", "Parametric loss fit (Approach 3)", ORANGE, "s", 0.2))
    for ap, label, col, mk, off in spec:
        xs, ys, lo, hi = [], [], [], []
        for i, conv in enumerate(CONVENTIONS):
            r = table[conv][ap]
            if not r:
                continue
            xs.append(i + off); ys.append(r["a"])
            ci = (table[conv].get("ci") or {}).get(ap)
            lo.append(r["a"] - ci[0] if ci else 0); hi.append(ci[1] - r["a"] if ci else 0)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=col, ms=7, capsize=3, lw=1.5, label=label)
    ax.set_xticks(range(len(CONVENTIONS))); ax.set_xticklabels(CONVENTIONS, fontsize=8)
    ax.set_ylabel("exponent $a$ in $N^*\\propto C^a$")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    _style(ax)
    fig.tight_layout(); fig.savefig(path + ".png", dpi=200); fig.savefig(path + ".pdf"); plt.close(fig)


def fig_shrink(rows, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for ap, label, col, mk in (("a1", "Approach 1", AQUA, "D"), ("a2", "Approach 2", BLUE, "o"), ("a3", "Approach 3", ORANGE, "s")):
        pts = [(r["n_runs"], r[f"{ap}_a_spread"]) for r in rows if r[f"{ap}_a_spread"] != ""]
        if pts:
            ax.plot(*zip(*pts), mk, color=col, ms=7, mec="white", mew=0.8, label=label)
    ax.set_xlabel("number of runs in the fitting grid")
    ax.set_ylabel("exponent spread across conventions")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    _style(ax)
    fig.tight_layout(); fig.savefig(path + ".png", dpi=200); fig.savefig(path + ".pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot_conv", default="total", choices=CONVENTIONS)
    ap.add_argument("--skip_shrink", action="store_true")
    ap.add_argument("--skip_perturb", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.runs)
    if len(runs) < 6:
        raise SystemExit(f"only {len(runs)} finished runs in {a.runs}/ - need a few more")
    rng = np.random.default_rng(a.seed)
    budgets = sorted({r["budget"] for r in runs})
    c_ref = math.exp(np.mean(np.log(budgets)))
    print(f"{len(runs)} runs, budgets {[f'{b:.0e}' for b in budgets]}, reference budget C_ref = {c_ref:.2e}")

    table = analyse(runs, runs, c_ref, boot=a.boot, rng=rng)
    sp = spreads(table)
    print(f"\n{'fit':<6}{'convention':<20}{'a':>7}{'D* (M tok)':>12}{'N* (M)':>9}{'D*/N*':>8}")
    for apn in ("a1", "a2", "a3"):
        for conv in CONVENTIONS:
            r = table[conv][apn]
            if r:
                print(f"{apn:<6}{conv:<20}{r['a']:>7.3f}{r['D_ref']/1e6:>12.0f}{r['N_ref']/1e6:>9.1f}{r['ratio_ref']:>8.1f}")
            else:
                print(f"{apn:<6}{conv:<20}  (fit failed: too few valid budgets/points)")
    print("\nspread across conventions:", json.dumps(sp, indent=1))
    pairs = (("a1", "a2"), ("a1", "a3"), ("a2", "a3"))
    gap = {c: {f"{x}_vs_{y}": ({"a_gap": table[c][x]["a"] - table[c][y]["a"],
                               "D_rel_gap": (table[c][x]["D_ref"] - table[c][y]["D_ref"]) / table[c][y]["D_ref"]}
                              if table[c][x] and table[c][y] else None) for x, y in pairs} for c in CONVENTIONS}
    print("pairwise approach gaps (RQ3):", json.dumps(gap, indent=1))

    shrink = [] if a.skip_shrink else shrink_sweep(runs, c_ref)
    if shrink:
        with open(os.path.join(a.out, "shrink_sweep.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(shrink[0].keys()))
            w.writeheader(); w.writerows(shrink)
        fig_shrink(shrink, os.path.join(a.out, "fig_shrink"))
    strip = lambda t: {c: {k: ({kk: vv for kk, vv in v.items() if kk not in ("pts", "theta")} if isinstance(v, dict) and k != "ci" else v)
                            for k, v in row.items()} for c, row in t.items()}
    summary = dict(c_ref=c_ref, budgets=budgets, table=strip(table), spreads=sp, approach_gap=gap,
                   compute=compute_report(runs))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    if not a.skip_perturb:
        prt = perturbation_sweep(runs, c_ref, a.seed)
        with open(os.path.join(a.out, "perturbation_sweep.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(prt[0].keys()))
            w.writeheader(); w.writerows(prt)
        fig_perturbation(prt, os.path.join(a.out, "fig_perturbation"))
        print("\nperturbations of N (base convention: " + PERTURB_BASE + "); mean over draws for noise")
        print(f"{'kind':<7}{'level':>10}{'a (A2)':>9}{'a (A3)':>9}{'tok/par A2':>12}{'tok/par A3':>12}{'D* A3 (M)':>11}")
        seen = {}
        for r in prt:
            seen.setdefault((r["kind"], r["level"]), []).append(r)
        mean = lambda rs, k: (np.mean([x[k] for x in rs if x[k] != ""]) if any(x[k] != "" for x in rs) else float("nan"))
        for (kind, lv), rs in seen.items():
            print(f"{kind:<7}{lv:>10.3g}{mean(rs,'a2_a'):>9.3f}{mean(rs,'a3_a'):>9.3f}{mean(rs,'a2_ratio_ref'):>12.2f}"
                  f"{mean(rs,'a3_ratio_ref'):>12.2f}{mean(rs,'a3_D_ref')/1e6:>11.0f}")
    write_latex(table, os.path.join(a.out, "table_conventions.tex"))
    fig_isoflop(runs, a.plot_conv, table[a.plot_conv]["a2"], os.path.join(a.out, "fig_isoflop"))
    fig_exponents(table, os.path.join(a.out, "fig_exponents"))
    print("\ncompute report (for NeurIPS checklist item 8):", json.dumps(summary["compute"], indent=1))
    print(f"wrote {a.out}/ (summary.json, table_conventions.tex, fig_*.png/pdf" + (", shrink_sweep.csv)" if shrink else ")"))


if __name__ == "__main__":
    main()
