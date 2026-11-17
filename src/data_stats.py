"""Per-image statistics for the competition training data -> data/meta/image_stats.csv

Usage: python src/data_stats.py data/extracted data/meta/image_stats.csv
"""
import os, sys
from multiprocessing import Pool

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def one(args):
    root, source, cid = args
    if source == "supplemental":
        ip, mp, ap = f"{root}/supplemental_images/{cid}.png", f"{root}/supplemental_masks/{cid}.npy", None
    elif source == "forged":
        ip, mp = f"{root}/train_images/forged/{cid}.png", f"{root}/train_masks/{cid}.npy"
        ap = f"{root}/train_images/authentic/{cid}.png"
    else:
        ip, mp, ap = f"{root}/train_images/authentic/{cid}.png", None, None
    im = Image.open(ip)
    r = dict(case_id=cid, source=source, width=im.size[0], height=im.size[1], mode=im.mode,
             file_kb=os.path.getsize(ip) // 1024)
    rgb = im.convert("RGB")
    r["phash"] = str(imagehash.phash(rgb, hash_size=16))
    g = np.asarray(rgb.convert("L"))
    r["gray_frac"] = float((np.asarray(rgb).std(axis=2) < 3).mean())  # near-grayscale pixel share
    if mp:
        m = np.load(mp)
        r["mask_shape_ok"] = m.shape[1:] == g.shape
        r["n_inst"] = m.shape[0]
        r["mask_frac"] = float(m.any(0).mean())
        r["inst_overlap_px"] = int((m.sum(0) > 1).sum())
        r["comps_per_inst"] = ";".join(str(cv2.connectedComponents(x)[0] - 1) for x in m)
        if ap and os.path.exists(ap):
            a = Image.open(ap).convert("RGB")
            r["twin_same_size"] = a.size == rgb.size
            if a.size == rgb.size:
                d = np.abs(np.asarray(a).astype(int) - np.asarray(rgb).astype(int)).max(2) > 8
                u = m.any(0)
                r["diff_frac"] = float(d.mean())
                r["diff_in_mask"] = float((d & u).sum() / max(d.sum(), 1))   # share of changed px inside mask
                r["mask_changed"] = float((d & u).sum() / max(u.sum(), 1))   # share of mask px that changed
    return r


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    jobs = [(root, "authentic", f[:-4]) for f in os.listdir(f"{root}/train_images/authentic")]
    jobs += [(root, "forged", f[:-4]) for f in os.listdir(f"{root}/train_images/forged")]
    jobs += [(root, "supplemental", f[:-4]) for f in os.listdir(f"{root}/supplemental_images")]
    with Pool(4) as p:
        rows = p.map(one, jobs, chunksize=16)
    pd.DataFrame(rows).sort_values(["source", "case_id"]).to_csv(out, index=False)
    print(len(rows), "rows ->", out)
