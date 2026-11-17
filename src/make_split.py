"""Create the one fixed train/dev/test split -> data/meta/split.csv  (run once; do not re-run to reshuffle).

Rows are evaluation units (one image each):
  uid = a_<id> (train authentic), f_<id> (train forged), s_<id> (supplemental; ids clash with train ids
  but are different images, hence the prefix).

Grouping (a group never crosses splits):
  1. authentic/forged twins that share a case_id;
  2. near-duplicates across ids: 256-bit pHash Hamming distance <= 48 on the base image
     (threshold chosen by eye: <=~44 were the same base image, >=~56 were merely the same dataset).
There is no paper/DOI metadata in the competition files, so "same paper" cannot be grouped.

Stratified by (source, image type). Synthetic train groups: 60/20/20. Supplemental (real figures, n=48):
50/50 dev/test, none in train.
"""
import json, sys

import numpy as np
import pandas as pd

SEED = 20261117
NEAR_DUP = 48
root_meta = "data/meta"

st = pd.read_csv(f"{root_meta}/image_stats.csv", dtype={"case_id": str})
st["uid"] = st.source.map({"authentic": "a_", "forged": "f_", "supplemental": "s_"}) + st.case_id
types = pd.read_csv(f"{root_meta}/image_types.csv", dtype={"case_id": str})
tmap = {(r.source, r.case_id): r.image_type for r in types.itertuples()}
st["image_type"] = [tmap.get(("forged" if s == "authentic" else s, c), "none") for s, c in zip(st.source, st.case_id)]

# union-find over base ids
au = set(st[st.source == "authentic"].case_id)
base = pd.concat([st[st.source == "authentic"], st[(st.source == "forged") & ~st.case_id.isin(au)],
                  st[st.source == "supplemental"]])
H = np.array([[int(b) for b in bin(int(h, 16))[2:].zfill(256)] for h in base.phash], dtype=np.uint8)
D = np.zeros((len(H), len(H)), dtype=np.int32)
for i in range(len(H)):
    D[i] = (H != H[i]).sum(1)
key = [("s_" if s == "supplemental" else "t_") + c for s, c in zip(base.source, base.case_id)]
parent = {k: k for k in key}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for i, j in zip(*np.where(np.triu(D <= NEAR_DUP, 1))):
    parent[find(key[i])] = find(key[j])
st["group"] = [find(("s_" if s == "supplemental" else "t_") + c) for s, c in zip(st.source, st.case_id)]

g = st.groupby("group").agg(real=("source", lambda s: (s == "supplemental").any()),
                            image_type=("image_type", lambda s: s.mode().iat[0]), n=("uid", "size")).reset_index()
rng = np.random.default_rng(SEED)
g["split"] = ""
for (real, _), idx in g.groupby(["real", "image_type"]).groups.items():
    idx = rng.permutation(np.array(sorted(idx)))
    n = len(idx)
    if real:
        cut = [0, (n + 1) // 2]            # dev, test
        names = ["dev", "test"]
        parts = [idx[: cut[1]], idx[cut[1]:]]
    else:
        a, b = int(round(n * 0.6)), int(round(n * 0.8))
        names, parts = ["train", "dev", "test"], [idx[:a], idx[a:b], idx[b:]]
    for nm, p in zip(names, parts):
        g.loc[p, "split"] = nm
st = st.merge(g[["group", "split"]], on="group")
st["label"] = np.where(st.source == "authentic", "authentic", "forged")
st["domain"] = np.where(st.source == "supplemental", "real_figure", "synthetic")
out = st[["uid", "case_id", "source", "domain", "label", "image_type", "group", "split", "width", "height"]].sort_values("uid")
assert out.groupby("group").split.nunique().max() == 1
out.to_csv(f"{root_meta}/split.csv", index=False)
print(pd.crosstab([out.domain, out.label, out.image_type], out.split, margins=True))
