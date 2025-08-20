#!/usr/bin/env python3
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if len(sys.argv) < 3:
    print("Usage: python eval_coco_json.py <instances_val2017.json> <detections.json>")
    raise SystemExit(1)

ann, dt = sys.argv[1], sys.argv[2]
coco = COCO(ann)
cocoDt = coco.loadRes(dt)
e = COCOeval(coco, cocoDt, iouType='bbox')
e.evaluate(); e.accumulate(); e.summarize()

print("==== SUMMARY ====")
print(f"mAP@[.50:.95]: {e.stats[0]:.4f}")
print(f"AP50:          {e.stats[1]:.4f}")
print(f"AR@100:        {e.stats[8]:.4f}")
