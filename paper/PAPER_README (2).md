# Paper folder

Draft LaTeX for the NeurIPS-style write-up. It contains **no results**: every `\TODO{...}` must be replaced
with text backed by `results/summary.json`, `results/shrink_sweep.csv` or the figures, or deleted.

1. Download the official template from https://neurips.cc/public/guides/PaperChecklist (the style-file link
   in the first paragraph) and upload the zip to Overleaf as a new project.
2. Copy `sections/`, `tables/` and `references.bib` into the project. After `python fit.py --boot 200`,
   upload everything in the repo's `results/` folder as a folder named `results`.
3. Follow `main_snippet.tex` to wire the sections into the template's main file. Keep the template's checklist.
4. Fill the checklist using `checklist_answers.md`.
5. Search the project for `TODO` before exporting the PDF (the `\TODO` macro prints in red so it is easy to see).

Things to verify yourself: author lists and venues in `references.bib`; the description of Schaeffer et al.'s three
interpretations and four perturbations in Sections 1 and 3 of the draft (checked against their abstract and the
parts of the paper I could read; confirm against the PDF); numbers quoted in the appendix recovery test.
