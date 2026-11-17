# Phase 0 report

## Hardware
No GPU. 4 CPU cores, 15 GB RAM, 19 GB free disk after setup. The container is ephemeral; only what is
pushed to git survives (data and weights must be re-downloaded in a new session).

## Forgeryscope (commit 9fdc66e, package v1.0.2)

**License.** Code and the author's part of five checkpoints (four Next-ViT embedders, the ALIKED+LightGlue
blot matcher): MIT — use, modify, redistribute, commercial use allowed, keep the notice. Upstream parts keep
their terms (Next-ViT Apache-2.0, ALIKED BSD-3-Clause, LightGlue Apache-2.0). The two YOLO detectors
(`yolo_panel_extractor.pt`, `yolo_lane_extractor.pt`) are **not** MIT: Ultralytics AGPL-3.0 by default, or a
paid Enterprise licence. For a later antibody-vendor tool this matters: shipping or serving the YOLO parts
means AGPL obligations or a licence, or replacing the two detectors.

**Weights.** Nothing in git; all seven checkpoints download from the GitHub release `models-v1`. All seven
downloaded here and match the SHA256 values in `model_zoo.py`. LightGlue is an extra install from GitHub.

**Missing compared to the winning submission** (per SOLUTION.md and the code):
- the public Kaggle segmentation kernel used as a fallback when nothing matched (author: about +0.005–0.01);
- two of the three microscopy embedders (ensemble reduced to one);
- the second YOLO panel detector (ensemble of two, author: +0.002–0.003);
- all training code, the training data (PMC crops), and the cleaned BioFors-based validation set;
- a batch/submission script: the "simplified Kaggle submission" notebook only runs two example images.
SOLUTION.md gives no score for the simplified release. Its numbers are leaderboard scores of the full
solution on the hidden test set: 0.456–0.458 public, 0.450 private for the submission matching this config.

## Metric
`eval/recodai_f1.py` is the organizers' notebook `metric/recodai-f1`, copied verbatim. 12 tests pass,
including the four requested and the notebook's own five doctests. Behaviours worth knowing:
- authentic image: 1 if predicted "authentic", else 0; forged image predicted "authentic": 0;
- forged image: predicted instances are matched one-to-one to true instances (Hungarian), pixel F1 per
  match, averaged over true instances, then multiplied by n_true / max(n_pred, n_true) — so extra
  predicted instances are penalised, and merging two groups into one mask costs a lot;
- final score is the plain mean over images.

## Data
See `docs/data_report.md`. Headline: train is synthetic single-panel copy-move; only the 48 supplemental
images are real multi-panel figures like the hidden test set.

## Open points
1. A local dev score will be dominated by synthetic images Forgeryscope was not designed for (the author
   says he skipped duplicated-cell forgeries). Reproducing ~0.45 locally is not possible from this data;
   only the real-figure slice (25 dev images, 7 of them blots) is comparable in kind, and it has no
   authentic images, so false alarms on real figures cannot be measured at all.
2. No GPU: Forgeryscope runs on CPU (panel detector about 0.5 s/image); training anything is impractical here.
3. Kaggle access works through a saved network secret; the CLI needs placeholder legacy credentials.
