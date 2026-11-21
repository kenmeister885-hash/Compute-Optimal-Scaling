"""Check that fit.py recovers a known scaling law.  Run:  python tests/test_fit_recovery.py"""
import math
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fit  # noqa: E402
from make_synthetic import A, AL, B, BE, E, make  # noqa: E402


def main():
    d = tempfile.mkdtemp()
    try:
        make(d, [3e15, 1e16, 3e16, 1e17], noise=0.005)
        runs = fit.load_runs(d)
        budgets = sorted({r["budget"] for r in runs})
        c_ref = math.exp(np.mean(np.log(budgets)))
        lad = fit.ladder_fit(runs, "total")
        truth = [math.log(A), math.log(B), math.log(E), AL, BE]
        Nt, Dt = fit.param_opt(truth, c_ref, lad)
        at = (math.log(fit.param_opt(truth, 2 * c_ref, lad)[0]) - math.log(fit.param_opt(truth, c_ref / 2, lad)[0])) / (2 * math.log(2))
        print(f"{len(runs)} runs | TRUE: a={at:.3f}  N*={Nt/1e6:.1f}M  D*={Dt/1e6:.0f}M")
        r1 = fit.approach1(runs, "total", c_ref)
        r2 = fit.approach2(runs, "total", c_ref)
        r3 = fit.approach3(runs, "total", c_ref, lad)
        ok = True
        # approach 1 picks among DISCRETE model sizes, so its tolerance is looser
        for name, r, tol_a, tol_D in (("approach1", r1, 0.15, 0.40), ("approach2", r2, 0.10, 0.25), ("approach3", r3, 0.05, 0.15)):
            if r is None:
                print(f"{name}: FAILED to fit"); ok = False; continue
            ea, eD = abs(r["a"] - at), abs(r["D_ref"] / Dt - 1)
            print(f"{name}: a={r['a']:.3f} (err {ea:.3f}, tol {tol_a})  D*={r['D_ref']/1e6:.0f}M (rel err {eD:.2f}, tol {tol_D})")
            ok &= ea < tol_a and eD < tol_D
        if r3:
            print(f"approach3 recovered alpha={r3['alpha']:.2f} (true {AL}), beta={r3['beta']:.2f} (true {BE})")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    sys.exit(main())
