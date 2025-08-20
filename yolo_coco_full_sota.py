#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO val2017 全量推理（YOLOv8/YOLO11），导出 COCO detections JSON（含二次NMS去重）
- 支持 TTA、批量推理、跨类/类内二选一NMS、可选可视化（独立可视化阈值）
- 推荐：yolo11x.pt / imgsz=1280 / conf=0.001 / iou=0.7 / tta
"""

import os, json, argparse, math
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO

# ------------- 工具：IoU & NMS（xyxy） -------------
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: [x1,y1,x2,y2]
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    # 朴素贪心NMS；返回保留索引
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest], dtype=np.float32)
        order = rest[ious <= iou_thr]
    return keep

# ------------- COCO 映射 & 可视化 -------------
def load_coco_index(ann_path: str):
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    fname2id = {im["file_name"]: int(im["id"]) for im in ann["images"]}
    name2id = {c["name"].strip().lower(): int(c["id"]) for c in ann["categories"]}
    return fname2id, name2id

ALIASES = {
    "wine glass":"wine glass","sports ball":"sports ball",
    "dining table":"dining table","potted plant":"potted plant",
    "hair drier":"hair drier","cell phone":"cell phone",
}

def draw_boxes(img_path: str, dets: List[dict], vis_path: str, vis_conf: float):
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for d in dets:
        if d["score"] < vis_conf: 
            continue
        x,y,w,h = d["bbox"]; x2, y2 = x+w, y+h
        cid = d.get("category_id", 0)
        color = (50 + (cid*37)%180, 50 + (cid*81)%180, 50 + (cid*113)%180)
        for t in range(3):
            draw.rectangle([x+t,y+t,x2-t,y2-t], outline=color, width=1)
        lab = d.get("label", str(cid)); txt = f"{lab} {d['score']:.2f}"
        th = getattr(font, "size", 16); tw = draw.textlength(txt, font=font)
        draw.rectangle([x, y-th-2, x+tw+6, y], fill=color)
        draw.text((x+3, y-th-1), txt, fill=(0,0,0), font=font)
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    im.save(vis_path, quality=95)

# ------------- 主流程 -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val-img", required=True)
    ap.add_argument("--val-ann", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou",  type=float, default=0.7)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--agnostic", action="store_true", help="额外NMS是否跨类；默认 False=类内NMS")
    ap.add_argument("--extra-nms", type=float, default=0.7, help="二次全局NMS阈值")
    ap.add_argument("--max-det", type=int, default=300)

    ap.add_argument("--vis-dir", default=None)
    ap.add_argument("--vis-conf", type=float, default=0.35)
    args = ap.parse_args()

    fname2id, name2id = load_coco_index(args.val_ann)

    # 收集所有图片路径（严格以标注为准，确保全量5000张）
    img_files = []
    for fn in sorted(os.listdir(args.val_img)):
        if fn.lower().endswith((".jpg",".jpeg",".png")) and fn in fname2id:
            img_files.append(os.path.join(args.val_img, fn))
    print(f"[Info] images: {len(img_files)}")

    model = YOLO(args.model)
    # 一次性批量预测（Ultralytics会自动分批）
    res = model.predict(
        source=img_files,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=False,
        verbose=False,
        stream=False,
        augment=args.tta,
        agnostic_nms=args.agnostic,  # 先让内部NMS可跨类
        max_det=args.max_det,
        batch=args.batch,
    )

    results_total: List[dict] = []

    for i, r in enumerate(res, 1):
        # 文件名 -> image_id
        fn = os.path.basename(r.path)
        image_id = fname2id.get(fn)
        dets_img: List[dict] = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names  # {id: name}

            # 组装为列表（先用内部NMS的输出）
            per_class: Dict[int, List[int]] = {}
            for j in range(xyxy.shape[0]):
                name = names.get(int(cls[j]), "").strip().lower()
                name = ALIASES.get(name, name)
                if name not in name2id:
                    continue
                dets_img.append({
                    "xyxy": xyxy[j],
                    "score": float(conf[j]),
                    "label": name,
                    "category_id": int(name2id[name]),
                })

            # 二次NMS：类内 or 跨类
            if dets_img:
                boxes = np.stack([d["xyxy"] for d in dets_img], axis=0)
                scores = np.array([d["score"] for d in dets_img], dtype=np.float32)
                if args.agnostic:
                    keep_idx = nms_xyxy(boxes, scores, args.extra_nms)
                    dets_img = [dets_img[k] for k in keep_idx]
                else:
                    kept_all = []
                    # 按类别分别NMS
                    by_cls: Dict[int, List[int]] = {}
                    for idx, d in enumerate(dets_img):
                        by_cls.setdefault(d["category_id"], []).append(idx)
                    for cid, idxs in by_cls.items():
                        b = boxes[idxs]
                        s = scores[idxs]
                        kk = nms_xyxy(b, s, args.extra_nms)
                        kept_all.extend([idxs[k] for k in kk])
                    kept_all = sorted(set(kept_all), key=lambda k: scores[k], reverse=True)
                    dets_img = [dets_img[k] for k in kept_all]

        # 写入 COCO JSON（xyxy->xywh）
        for d in dets_img:
            x1,y1,x2,y2 = d["xyxy"]
            x = float(x1); y = float(y1); w = float(x2 - x1); h = float(y2 - y1)
            results_total.append({
                "image_id": int(image_id),
                "category_id": int(d["category_id"]),
                "bbox": [x, y, w, h],
                "score": float(d["score"]),
                "label": d["label"]
            })

        # 可选可视化
        if args.vis_dir is not None:
            vis_path = os.path.join(args.vis_dir, fn)
            draw_boxes(os.path.join(args.val_img, fn), results_total[-len(dets_img):], vis_path, args.vis_conf)

        if (i <= 5) or (i % 100 == 0):
            print(f"[{i}/{len(img_files)}] {fn}: dets={len(dets_img)}  total={len(results_total)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results_total, f)
    print(f"[Save] COCO detections -> {args.out}  (items={len(results_total)})")
    print("[Done] 推理完成。现在可以用 COCOeval 直接评 5000 张。")

if __name__ == "__main__":
    main()
