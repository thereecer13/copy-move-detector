"""Assign one image type per labeled image from YOLO panel detections -> data/meta/image_types.csv

Rule (detections with conf >= 0.3):
  - forged/supplemental image: the panel label covering the most ground-truth mask pixels;
  - if no panel touches the mask: the label with the largest total box area; if no detections: "none".
Authentic train images inherit the type of their forged twin (same base image).
Also records n_panels, so single-panel vs multi-panel images can be separated.
"""
import json, sys

import numpy as np
import pandas as pd

root = "data/extracted"
rows = []
for line in open("data/meta/panels.jsonl"):
    r = json.loads(line)
    if r["source"] == "test":
        continue
    dets = [d for d in r["dets"] if d["conf"] >= 0.3]
    mp = f"{root}/train_masks/{r['case_id']}.npy" if r["source"] == "forged" else f"{root}/supplemental_masks/{r['case_id']}.npy"
    u = np.load(mp).any(0)
    cover, area = {}, {}
    for d in dets:
        x0, y0, x1, y1 = [int(round(v)) for v in d["box"]]
        cover[d["label"]] = cover.get(d["label"], 0) + int(u[y0:y1, x0:x1].sum())
        area[d["label"]] = area.get(d["label"], 0) + (x1 - x0) * (y1 - y0)
    if cover and max(cover.values()) > 0:
        t, how = max(cover, key=cover.get), "mask"
    elif area:
        t, how = max(area, key=area.get), "area"
    else:
        t, how = "none", "none"
    in_any = 0.0
    if dets:
        c = np.zeros_like(u)
        for d in dets:
            x0, y0, x1, y1 = [int(round(v)) for v in d["box"]]
            c[y0:y1, x0:x1] = True
        in_any = float((u & c).sum() / max(u.sum(), 1))
    rows.append(dict(source=r["source"], case_id=r["case_id"], image_type=t, how=how, n_panels=len(dets),
                     labels=";".join(sorted({d["label"] for d in dets})), mask_in_panels=round(in_any, 3)))
df = pd.DataFrame(rows)
df.to_csv("data/meta/image_types.csv", index=False)
print(pd.crosstab(df.image_type, df.source, margins=True))
print(df.groupby("source").n_panels.describe().round(1))
print("mask px inside any detected panel:", df.groupby("source").mask_in_panels.mean().round(3).to_dict())
