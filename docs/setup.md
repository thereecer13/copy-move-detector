# Setup (as done in Phase 0)

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision   # CPU build
git clone https://github.com/vlad3996/forgeryscope external/forgeryscope             # commit 9fdc66e
uv pip install -e external/forgeryscope "git+https://github.com/cvg/LightGlue.git" \
    kaggle six pytest scikit-learn scipy numba imagehash
```

Checkpoints: the seven files from
`https://github.com/vlad3996/forgeryscope/releases/download/models-v1/` into `models/forgeryscope/`
(SHA256 verified against `forgeryscope/model_zoo.py`).

Data: `kaggle competitions download -c recodai-luc-scientific-image-forgery-detection`, unzipped to
`data/extracted/`. In the cloud session the Kaggle token is injected by the network proxy; the CLI is
run with placeholder `KAGGLE_USERNAME`/`KAGGLE_KEY` (the 2.x CLI's `KAGGLE_API_TOKEN` path fails
because it validates the placeholder token itself).

Metadata: `python src/data_stats.py data/extracted data/meta/image_stats.csv`,
`python src/panel_types.py data/extracted data/meta/panels.jsonl`, then `python src/make_split.py` (once).
