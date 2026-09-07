# Adaptive Self-Consistency for Long-Form Factuality on Long-Tail Facts

[![KONVENS 2026](https://img.shields.io/badge/KONVENS-2026-004C97)](https://konvens2026.uni-hamburg.de/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![🤗 Dataset: RiDiC](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-RiDiC-orange)](https://huggingface.co/datasets/s-nlp/RiDiC_rivers_self_consistency_99_generations)

Code and paper for the KONVENS 2026 paper
Moskovskiy, D., Grigoryan, R., Shuranov, E., Yin, W., & Panchenko, A. (2026). *Adaptive self-consistency for long-form factuality on long-tail facts*. In *Proceedings of the Conference on Natural Language Processing (KONVENS 2026)*. Hamburg, Germany: German Society for Computational Linguistics and Language Technology.
Contact: `A.Panchenko@skol.tech`

## TL;DR

- **LLMAgg** (LLM-based aggregation) beats voting, selection, and a compute-matched best-of-K logprob selector, across 2 languages × 3 domains.
- Most of the factuality gain **saturates by K≈8–16** samples; almost nothing left to gain past K=20.
- A lightweight pre-generation **random-forest controller** retains **>99%** of full-budget (K=99) factual precision while cutting sampled generations by **2–29%**.
- Against a hand-calibrated K=20 reference, the same controller does **not** produce net savings — see the paper's sensitivity analysis for why.

## Abstract

> Long-form question answering for rare entities remains challenging for large language models, while inference-time methods for improving factual precision in this setting are underexplored. We study multi-sample self-consistency for long-form generation across various Wikipedia entity popularity tiers using the RiDiC dataset (Braslavski et al., 2026), using fact-level evaluation. We compare several aggregation strategies and find that LLM-based aggregation consistently outperforms voting and selection baselines, including a compute-matched best-of-K logprob selector, across languages and domains, with most gains achieved at relatively small sample budgets. Motivated by this early-saturation effect, we introduce an adaptive policy that predicts an appropriate generation budget from pre-generation features. Against a conservative full-budget self-consistency reference (99 generations per question), the adaptive policy retains over 99% of factual precision while reducing the number of sampled generations by 2–29% across various language–domain pairs. We further investigate how these savings depend on the reference budget and identify the regime where adaptive routing helps most.

## Pipeline overview

*Query → self-consistency sampling → LLM aggregation → final answer, with the adaptive controller predicting the sampling budget K and FactOWL scoring the output for fact-level precision.*

## Repository layout

| Path | Contents |
|---|---|
| [`latex/`](latex/) | Paper source (`acl_latex.tex`), KONVENS beamer deck (`adaptive_self_consistency_beamer.tex`), figures |
| [`code/`](code/) | Minimal reproduction package: FactOWL scoring, LLMAgg budget curves, adaptive controller, analysis scripts that build every paper table and figure |

## Data

Candidate pools are the public HF exports
[`s-nlp/RiDiC_rivers_self_consistency_99_generations`](https://huggingface.co/datasets/s-nlp/RiDiC_rivers_self_consistency_99_generations),
[`s-nlp/RiDiC_cars_self_consistency_99_generations`](https://huggingface.co/datasets/s-nlp/RiDiC_cars_self_consistency_99_generations),
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

## Citation

```bibtex
@inproceedings{moskovskiy2026adaptive,
  title     = {Adaptive Self-Consistency for Long-Form Factuality on Long-Tail Facts},
  author    = {Moskovskiy, Daniil and Grigoryan, Rafael and Shuranov, Evgenii and Yin, Wenshuai and Panchenko, Alexander},
  booktitle = {Proceedings of the 22nd Conference on Natural Language Processing (KONVENS 2026)},
  year      = {2026},
  address   = {Hamburg, Germany},
  publisher = {German Society for Computational Linguistics and Language Technology},
  note      = {To appear. BibTeX entry will be updated with final proceedings details (pages, DOI, arXiv ID).}
}
```
