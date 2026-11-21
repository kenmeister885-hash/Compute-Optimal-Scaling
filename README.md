# Robustness of Chinchilla compute-optimal scaling at small scale

Small-scale replication of the parameter-counting robustness analysis in Schaeffer et al. (2025),
*Evaluating the Robustness of Chinchilla Compute-Optimal Scaling* (arXiv:2509.23963), on GPT-style
models trained on C4. Models: 7 sizes (0.4M-59M non-embedding params), IsoFLOP grid at 3-4 compute
budgets (3e15-1e17 FLOPs).

## Set-up (single GPU, e.g. RTX 4070)
```bash
Install PyTorch for your CUDA version from https://pytorch.org, then: pip install numpy scipy matplotlib tiktoken datasets
python prepare_data.py --train_tokens 350e6   # slowest step (30-60+ min); bottleneck is streaming C4, needs internet
python bench.py --micro_bsz 16           # a few minutes: measures tokens/s per model size on YOUR GPU
python grid.py --preset fast --micro_bsz 16            # prints the plan + time estimate
python grid.py --preset fast --micro_bsz 16 --run      # trains the grid; finished runs are skipped
python fit.py --boot 200                 # tables, figures, bootstrap CIs, shrink sweep -> results/
python tests/test_fit_recovery.py        # sanity check of the fitting code on a known synthetic law
```
If you hit CUDA out-of-memory add `--micro_bsz 16` to `bench.py` / `grid.py`.
Cluster: `python grid.py --shard 0/4 --run` (one process per GPU, shards 0..3).
Presets: `--preset fast` (default, budgets 3e15/1e16/3e16) or `--preset full` (+1e17, needs >=480M tokens).

## Layout
- `scaling/model.py` model, **parameter-count conventions** (`count_params`), `scaling/sizes.py` size ladder + FLOPs formula
- `train.py` one run (own warmup+cosine schedule, data seen once, no repeats); `grid.py` planner/runner; `bench.py` throughput
- `fit.py` Approaches 1-3, conventions, bootstrap, shrink sweep, figures
- `results/summary.json`, `table_conventions.tex`, `shrink_sweep.csv`, `fig_*.pdf` are paper-ready

## Known limitations / things to align before submitting
- **Parameter-count conventions** (`CONVENTIONS`: total / non-embedding / non-embedding+head) are ours and differ from
  Schaeffer et al.'s three interpretations (three ways of computing Chinchilla's reported sizes, differing by <=15.2%).
  Their four structured perturbations of N (multiplicative, additive, systematic bias, log-normal noise) are applied by
  `fit.py` to the total-parameter convention -> `results/perturbation_sweep.csv`, `fig_perturbation.pdf`.
- All three Hoffmann approaches are implemented (1 = training-curve envelope, 2 = IsoFLOP parabolas, 3 = parametric loss).
  Approach 1 picks the best run by loss, which does not depend on the N convention, so its D* is identical across
  conventions by construction (only its N and exponent change). Its N* is a step function over the discrete size ladder,
  and its intermediate points come from runs whose LR schedule has not finished decaying (as in Hoffmann et al.).
  `train.py` now logs ~8 intermediate validation losses per run (cheap, 10 batches each) for Approach 1.
- Learning rates follow an **untuned** heuristic (`train.py:default_lr`); per-size LR tuning is not done.
- Vocabulary is 50k, so for the smallest models the LM head dominates parameters and FLOPs. D* (optimal token count)
  is comparable across conventions; the exponent `a` for non-embedding N is not expected to match total-N's.
- Approach 3 evaluates optima at C_ref (geometric mean of the budgets) using FLOPs/token as a function of N.
- Smoke-tested on CPU with synthetic data; real-data behaviour is untested until you run the pilot.

