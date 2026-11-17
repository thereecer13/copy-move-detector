# copy-move-detector

Copy-move forgery detection for scientific figures, starting from the winning
solution of the Kaggle competition *Recod.ai/LUC – Scientific Image Forgery
Detection* (Forgeryscope) and improving it, with a focus on western blots.

## Layout

| Path | What | In git? |
| --- | --- | --- |
| `src/` | pipeline and data scripts | yes |
| `eval/` | official metric (verbatim copy), tests, eval harness | yes |
| `experiments/` | per-experiment outputs; `experiments.md` is the log | yes (small files only) |
| `notebooks/` | exploration, Kaggle submission notebook | yes |
| `data/meta/` | split, per-image stats, image types (no pixels) | yes |
| `data/extracted/` | competition data (CC BY-SA 4.0, no redistribution under the rules) | no |
| `models/` | downloaded checkpoints | no |
| `external/` | Forgeryscope clone, metric notebook, competition pages | no |

## Rules of the project

1. `data/meta/split.csv` is the one fixed split. Tune on `dev` only.
2. Every run on `test` is appended to `experiments/test_runs.log`.
3. Scores come from `eval/recodai_f1.py` (official metric, unmodified); `pytest eval` must pass.
4. One change per experiment; each gets a row in `experiments.md`.

## Setup

See `docs/setup.md`. Reports: `docs/phase0_report.md`, `docs/data_report.md`.
