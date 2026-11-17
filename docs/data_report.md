# Data report (Phase 0)

Source: `kaggle competitions download -c recodai-luc-scientific-image-forgery-detection` (5.2 GB zip, 8.9 GB unzipped).
Numbers below come from `data/meta/image_stats.csv`, `image_types.csv` and `split.csv`.

## What is in the download

| Folder | Files | Notes |
| --- | ---: | --- |
| `train_images/authentic/` | 2,377 PNG | no mask |
| `train_images/forged/` | 2,751 PNG | each has `train_masks/<id>.npy` |
| `supplemental_images/` + `supplemental_masks/` | 48 + 48 | all forged; added after launch |
| `test_images/` | 1 PNG (`45.png`) | placeholder; the real test set (~1,100 public + about as many private images) is hidden |
| `sample_submission.csv` | `case_id,annotation` | |

Labeled images: 5,176 (2,377 authentic, 2,799 forged). There are no authentic real figures.

## The most important finding: two very different kinds of image

**Train images are synthetic, single-panel forgeries.** All 2,377 authentic images have a forged twin
with the same id and size; the twin differs from it only inside the mask (median 96% of changed pixels
are inside the mask, and about half the mask changes, i.e. the pasted copy). They are single panels
(82% get at most one detected panel) such as a field of cells or one blot strip, with an object (a
cell, a band) copied and pasted *within* that panel. 374 forged images have no twin under their own
id; some are forgeries of an authentic image stored under another id.

**Supplemental images are real multi-panel figures** (median 12 detected panels, up to 36), annotated
with duplicated or overlapping panels and blot regions. This is what the description says the hidden
test set is ("confirmed forgeries pulled from retracted papers"), and what Forgeryscope was built for.
There are only 48 of them.

I could not verify what share of the hidden test set looks like which kind; the only visible test
image (45.png) is a real multi-panel blot figure.

## Mask format

`.npy`, `uint8`, shape `(n_instances, H, W)`, values 0/1, same H×W as the image (100%). One channel is
one duplication *group*: the source region and its copies together (usually 2 connected components,
sometimes 3–5). Instances never overlap. Instances per forged image: 1 → 1,902; 2 → 710; 3 → 176;
4+ → 11 (max 13, in a supplemental figure).

Mask area as a share of the image: train median 2.7% (10th–90th pct 0.4%–14%); supplemental median 6.7%.

## Image sizes

| | n | median W×H | range | megapixels (median / 90th pct) |
| --- | ---: | --- | --- | --- |
| train (authentic and forged alike) | 5,128 | 696×512 | 74×64 … 3888×3888 | 0.33 / 1.9 |
| supplemental | 48 | 1432×1251 | 384×381 … 4000×4417 | 1.75 / 10 |

Modes: RGB 3,426, RGBA 1,438, grayscale 312. A handful of sizes dominate train (1000×666, 256×256,
1600×1200, 696×520 …), which suggests it was built from a few public image collections.

## Rough breakdown by image type

Type = label from Forgeryscope's YOLO panel detector for the panel(s) under the mask (authentic images
inherit their twin's type). This is a model's opinion, not ground truth; in particular "Body Imaging"
here is a catch-all that includes macro photos (e.g. corn cobs on a green background).

| Type | synthetic forged | synthetic authentic | real figures (supplemental) |
| --- | ---: | ---: | ---: |
| Microscopy | 1,759 | 1,500 | 34 |
| Western blot ("Blots") | 533 | 422 | 13 |
| Body Imaging | 401 | 397 | 1 |
| Graphs | 56 | 56 | 0 |
| no panel detected | 2 | 2 | 0 |

## The fixed split (`data/meta/split.csv`, seed 20261117)

Groups never cross splits. A group = authentic/forged twins + near-duplicates across ids (256-bit
perceptual hash, Hamming distance ≤ 48, threshold chosen by looking at pairs). 2,458 groups, 288 of them
span more than one id. The files carry no paper or DOI information, so "same paper" could not be grouped.
Images that merely come from the same source collection (similar look, different field of view) can
still be in different splits.

| | train | dev | test |
| --- | ---: | ---: | ---: |
| synthetic | 3,065 | 1,048 | 1,015 |
| real figures | 0 | 25 | 23 |

Stratified by domain × image type (see the script output for the full table). Real blot figures: 7 in
dev, 6 in test.
