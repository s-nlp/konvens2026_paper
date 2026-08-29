# Adaptive Self-Consistency for Long-Form Factuality on Long-Tail Facts

Code and paper for the KONVENS 2026 paper
*"Adaptive Self-Consistency for Long-Form Factuality on Long-Tail Facts"*
(Daniil Moskovskiy, Rafael Grigoryan, Evgenii Shuranov, Wenshuai Yin,
Alexander Panchenko — AIRI, Skoltech, MSU, ITMO, HSE).

Contact: `A.Panchenko@skol.tech`

## Repository layout

| Path | Contents |
|---|---|
| [`latex/`](latex/) | Paper source (`acl_latex.tex`), KONVENS beamer deck (`adaptive_self_consistency_beamer.tex`), figures |
| [`code/`](code/) | Minimal reproduction package: FactOWL scoring, LLMAgg budget curves, adaptive controller, analysis scripts that build every paper table and figure |

## Paper

Multi-sample self-consistency for long-form generation on RiDiC (3 domains ×
2 languages × Wikipedia-pageview tiers), fact-level evaluation with FactOWL.
LLM-based aggregation (LLMAgg) beats voting, selection, and a compute-matched
best-of-K logprob selector; budget gains saturate around K=8–16; a
pre-generation random-forest controller retains >99% of the full-budget (K=99)
factuality while saving 2–29% of sampled generations — with an explicit
analysis of why those savings do not transfer to a hand-calibrated K=20
reference.

Candidate pools are the public HF exports
[`s-nlp/RiDiC_rivers_self_consistency_99_generations`](https://huggingface.co/datasets/s-nlp/RiDiC_rivers_self_consistency_99_generations),
[`s-nlp/RiDiC_cars_self_consistency_99_generations`](https://huggingface.co/datasets/s-nlp/RiDiC_cars_self_consistency_99_generations)
and [`s-nlp/RiDiC_bad_weather_self_consistency_99_generations`](https://huggingface.co/datasets/s-nlp/RiDiC_bad_weather_self_consistency_99_generations)
(plus their Chinese counterparts).

## Reproducing the results

```bash
pip install -r code/requirements.txt
# stage 1 (GPU): FactOWL scoring + LLMAgg prefix curves
bash code/factowl/run_llmagg_curve_details_en.sh
bash code/factowl/run_llmagg_curve_details_zh.sh
# stage 2: adaptive controller
python code/factowl/train_llmagg_adaptivity_lang.py --lang en --retention 0.95
# stage 3 (CPU): paper tables and figures
python code/scripts/make_plots.py
```

Stage 1 needs a CUDA GPU (vLLM); stage 3 reads the saved per-topic TSVs from
the LongFT experiment checkout. Full pipeline description, inputs, expected
outputs and known gaps: [`code/README.md`](code/README.md).

## Building the paper and the deck

```bash
cd latex
pdflatex acl_latex.tex && bibtex acl_latex && pdflatex acl_latex.tex && pdflatex acl_latex.tex
xelatex adaptive_self_consistency_beamer.tex   # slides; add \def\showNotes for the notes build
```
