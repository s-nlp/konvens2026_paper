# code — minimal reproduction code for the paper results

Minimal code paths that produce the results of
**"Adaptive Self-Consistency for Long-Form Factuality on Long-Tail Facts"** (KONVENS 2026).

The package lives in the paper repository (`s-nlp/konvens2026_paper`) and works from any
nesting depth: stage-3 scripts and the shell wrappers locate the experiment checkout
(the directory containing `experiments/factowl_eval` — the LongFT checkout with the
saved TSVs) by walking up from their own location, falling back to `parents[2]`.
Set `OUT_DIR` (wrappers) to override the output target explicitly.

## Layout

```
code/
├── factowl/                     # stage 1–2: GPU scoring, aggregation curves, adaptive policy
│   ├── run_factowl_eval.py          # FactOWL atomic-fact scoring of candidate answers
│   ├── run_llmagg_curve.py          # LLMAgg prefix curves (k = 1..99)
│   ├── run_fact_majority_curve.py   # fact-majority voting curves
│   ├── analyze_sc_curves.py         # baseline curves from completion-score matrices
│   ├── plot_sc_curve_grid.py        # combined 3x2 SC-curve grid (stakeholder figure)
│   ├── train_llmagg_adaptivity.py   # per-domain adaptive controller
│   ├── train_llmagg_adaptivity_lang.py    # multilingual adaptive controller (paper)
│   ├── sweep_llmagg_adaptivity_lang_retention.py  # retention sweep 0.95 / 0.99
│   ├── run_llmagg_curve_details_en.sh     # reference batch: EN rivers/cars/disasters
│   ├── run_llmagg_curve_details_zh.sh     # reference batch: ZH rivers/cars/disasters
│   ├── factowl/                     # FactOWL scorer package (atomic facts + verification)
│   └── data/                        # prompts / demos used by the scorer
├── scripts/                     # stage 3: CPU analysis -> paper tables and figures
│   ├── robustness_analysis.py       # bootstrap CIs + k20/k99 gap
│   ├── targeted_robustness.py       # feature importance, failure profiles, difficulty profile
│   ├── reviewer_additions.py        # best-of-K logprob, cross-domain, catastrophic-failure audit
│   ├── train_adaptive_k20.py        # controller re-anchored to K=20
│   ├── reframe_adaptive_compute.py  # cost accounting (savings vs K=99 / K=20)
│   ├── reasc_investigation.py       # corrected post-hoc ReASC comparison
│   ├── evaluate_ridic_reasc.py      # ReASC evaluator (completion-level, post-fix)
│   └── make_plots.py                # paper figures from the TSVs
└── ReASC/                       # ReASC baseline package
```

## Environment

Python 3.11+ (original runs: 3.11/3.12). Install:

```bash
pip install -r requirements.txt
```

Stage 1 requires a CUDA GPU (originals ran on H200/A6000-class hardware with vLLM,
bf16, tensor parallelism 1, seed 0, max 256 new tokens; LLMAgg and FactOWL use
temperature 0). The 99-completion candidate pools are the public HF exports
(`s-nlp/RiDiC_{rivers,cars,bad_weather}_self_consistency_99_generations` and their
Chinese counterparts); their original decoding parameters are not recorded in the
exports, so exact reproduction of the pool means using the released pool itself.
RiDiC question metadata (titles, tiers) is read from the local HF datasets cache
(`~/.cache/huggingface/datasets/s-nlp___ri_di_c/...`).

## Stage 1 — candidate scoring and aggregation curves (GPU)

```bash
bash factowl/run_llmagg_curve_details_en.sh    # GPU=0, PYTHON=... overridable
bash factowl/run_llmagg_curve_details_zh.sh
```

Both wrappers accept environment overrides (`GPU`, `GPU_MEMORY_UTILIZATION`,
`K_LIST`, `MODEL`, `PYTHON`, `OUT_DIR`) and write per-topic detail/summary TSVs to
`experiments/factowl_eval/`. The K=20 rerun points are produced the same way with
`K_LIST=20` (originally archived under `paper_prep/experiments/llmagg_k20/`).
Fact-majority curves: `run_fact_majority_curve.py` (same CLI style).

## Stage 2 — adaptive controller (GPU-light; needs stage-1 detail TSVs)

```bash
python factowl/train_llmagg_adaptivity_lang.py --lang en --retention 0.95
python factowl/train_llmagg_adaptivity_lang.py --lang en --retention 0.99
python factowl/train_llmagg_adaptivity_lang.py --lang zh --retention 0.95
python factowl/train_llmagg_adaptivity_lang.py --lang zh --retention 0.99
python factowl/sweep_llmagg_adaptivity_lang_retention.py
```

Protocol: 300 trees, min leaf 3, five stratified shuffle splits, test 30%,
split seed 13, model seed 13 + fold index, retention targets 0.95/0.99,
pre-generation features only.

## Stage 3 — paper tables and figures (CPU)

Run from `code/scripts/` (each script writes into `../paper_prep/tables/`
or `../paper_prep/appendix/`):

| Script | Outputs (paper tables/figures) |
|---|---|
| `robustness_analysis.py` | `bootstrap_main_comparison`, `bootstrap_llmagg_prefix_curve`, `bootstrap_tier_comparison`, `llmagg_k20_k99_gap`, `bootstrap_adaptive_tradeoff` |
| `targeted_robustness.py` | `adaptive_feature_importance`, `adaptive_feature_ablation`, `adaptive_failure_profile`, `adaptive_failure_by_tier`, `domain_difficulty_profile` |
| `reviewer_additions.py` | `bestofk_logprob_k20`, `adaptive_cross_domain`, `adaptive_catastrophic_failure_analysis`, `reproducibility_statement.md` |
| `train_adaptive_k20.py` | `adaptive_k20_controller`, `adaptive_k20_controller_predictions` |
| `reframe_adaptive_compute.py` | `adaptive_cost_accounting`, `adaptive_cost_accounting_compact` |
| `reasc_investigation.py` | `reasc_comparison_k20`, appendix note + EN-cars provenance table |
| `make_plots.py` | `prefix_scaling_en_zh.pdf`, `adaptive_feature_importance.pdf`, `adaptive_tradeoff_scatter.pdf` |

Intermediate aggregations consumed by these scripts (`main_comparison_k20.tsv`,
`llmagg_prefix_curve_k1_20.tsv`, `tier_comparison_k20.tsv`,
`adaptive_retention_sweep_*.tsv`, `error_analysis_examples.tsv`, tier breakdowns)
are straightforward per-topic aggregations of the stage-1/2 outputs; the archived
copies used by the paper live in `paper_prep/tables/`.

## Known gaps

- Saved SeqProb selector artifacts (`seqprob_selection/*.csv`) exist for **EN only**;
  the ZH SeqProb cell of the main table predates this package and its run outputs
  were not persisted here.
- CLI defaults of the GPU entry scripts still carry the historical
  `/workspace/longft/...` paths; the shell wrappers and explicit flags shown above
  override them.
