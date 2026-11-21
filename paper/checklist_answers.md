# NeurIPS Paper Checklist: draft answers

Paste each answer and justification into the checklist that comes with the official 2026 template
(items taken from https://neurips.cc/public/guides/PaperChecklist, 16 questions). Change any answer
that stops being true once your results are in. Answers are visible to reviewers.

1. **Claims.** Yes. The abstract and introduction state a small-scale replication of a parameter-counting
   sensitivity analysis and list its limits. *Before submitting: re-read the abstract against the results.*
2. **Limitations.** Yes. Section "Limitations" (scale, embedding-dominated models, single seed, untuned
   hyperparameters, few data points, our own conventions, one dataset).
3. **Theory, assumptions and proofs.** NA. The paper has no theoretical results.
4. **Experimental result reproducibility.** Yes. Models, FLOPs accounting, training, fitting procedures and
   the grid are specified in Sections 3-4 and the appendix, and the code is released.
5. **Open access to data and code.** Yes. Anonymized repository with exact commands (README); data are the
   public C4 corpus prepared by `prepare_data.py`. *TODO: add the anonymized repo link or zip.*
6. **Experimental setting/details.** Yes. Section 3.3 and the appendix (hyperparameters, schedule, data split,
   learning-rate heuristic; note it is untuned).
7. **Experiment statistical significance.** Yes, with a caveat: bootstrap 95% intervals for the fitted
   quantities (Section 3.5), with the resampling unit stated. Training is done once per configuration, so
   seed variance is not captured (Limitations). *Only answer Yes if you ran `fit.py --boot`; otherwise answer
   No and say it was too costly or not done.*
8. **Experiments compute resources.** Yes. GPU model, per-run and total GPU-hours and FLOPs in the appendix
   ("Compute"), from `results/summary.json`. *Add pilot and failed runs by hand.*
9. **Code of ethics.** Yes. *Read https://neurips.cc/public/EthicsGuidelines first.*
10. **Broader impacts.** NA, or one sentence: foundational work on training-budget allocation with no direct
    path to harmful application.
11. **Safeguards.** NA. Small models trained from scratch; no high-risk model or scraped dataset is released.
12. **Licenses for existing assets.** Yes. Cite C4 (Raffel et al., 2020), tiktoken, PyTorch and the papers
    whose methods we follow. *TODO: look up and state each license and version (C4 dataset card on
    HuggingFace, tiktoken and PyTorch repositories); do not rely on this draft for the license names.*
13. **New assets.** Yes for the code (documented in the README). No datasets or model checkpoints are
    released. *Choose a license for your repo.*
14. **Crowdsourcing and human subjects.** NA.
15. **IRB approvals.** NA.
16. **Declaration of LLM usage.** *Decide this yourself.* NeurIPS only requires a declaration when an LLM is an
    important, original or non-standard part of the core methods. This project's code and drafts were
    written with an AI assistant (Claude); check your course's or venue's policy on disclosing AI assistance
    and state it accurately (for example in the checklist justification or an acknowledgements line).
