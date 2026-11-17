"""Run Forgeryscope's YOLO panel detector over labeled images and save every detection.

Used only to *describe* the data by image type (Blots / Microscopy / Graphs / Body Imaging /
Flow Cytometry). Usage: python src/panel_types.py data/extracted data/meta/panels.jsonl
"""
import json, os, sys

import numpy as np
from PIL import Image
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None
root, out = sys.argv[1], sys.argv[2]
model = YOLO("models/forgeryscope/yolo_panel_extractor.pt")
jobs = [("forged", f"{root}/train_images/forged/{f}") for f in sorted(os.listdir(f"{root}/train_images/forged"))]
jobs += [("supplemental", f"{root}/supplemental_images/{f}") for f in sorted(os.listdir(f"{root}/supplemental_images"))]
jobs += [("test", f"{root}/test_images/{f}") for f in sorted(os.listdir(f"{root}/test_images"))]
done = set()
if os.path.exists(out):
    done = {(json.loads(l)["source"], json.loads(l)["case_id"]) for l in open(out)}
with open(out, "a") as fh:
    for i, (src, p) in enumerate(jobs):
        cid = os.path.basename(p)[:-4]
        if (src, cid) in done:
            continue
        img = np.array(Image.open(p).convert("RGB"))
        r = model.predict(img, conf=0.25, iou=0.4, imgsz=640, device="cpu", verbose=False)[0]
        dets = [dict(label=model.names[int(b.cls)], conf=round(float(b.conf), 3),
                     box=[round(float(v), 1) for v in b.xyxy[0]]) for b in r.boxes]
        fh.write(json.dumps(dict(source=src, case_id=cid, h=img.shape[0], w=img.shape[1], dets=dets)) + "\n")
        fh.flush()
        if i % 200 == 0:
            print(i, len(jobs), flush=True)
