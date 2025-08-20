#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO val2017 全量推理（YOLOv8/YOLO11）- 流式版
- 目录流式读取（stream=True），避免一次性打开过多文件
- 二次 NMS（类内或跨类）进一步去重
- 可选可视化，使用 with 打开图片，及时释放句柄
- 默认参数适合最高精度：imgsz=1280, conf=0.001, iou=0.7, TTA
"""

import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO


# ---------------- IoU & NMS（xyxy） ----------------
def _iou_xyxy(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - y1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def _nms_xyxy(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([_iou_xyxy(boxes[i], boxes[j]) for j in rest], dtype=np.float32)
        order = rest[ious <= iou_thr]
    return keep


# ---------------- COCO 索引 & 可视化 ----------------
_ALIASES = {
    "wine glass": "wine glass",
    "sports ball": "sports ball",
    "dining table": "dining table",
    "potted plant": "potted plant",
    "hair drier": "hair drier",
    "cell phone": "cell phone",
}


def _load_coco_index(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    fname2id = {im["file_name"]: int(im["id"]) for im in ann["images"]}
    name2id = {c["name"].strip().lower(): int(c["id"]) for c in ann["categories"]}
    return fname2id, name2id


def _draw_boxes(img_path, dets, vis_path, vis_conf):
    # dets: [{"bbox":[x,y,w,h], "score":.., "label":.., "category_id":..}, ...]
    with Image.open(img_path) as _im:
        im = _im.convert("RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for d in dets:
        if d["score"] < vis_conf:
            continue
        x, y, w, h = d["bbox"]
        x2, y2 = x + w, y + h
        cid = d.get("category_id", 0)
        color = (50 + (cid * 37) % 180, 50 + (cid * 81) % 180, 50 + (cid * 113) % 180)
        for t in range(3):
            draw.rectangle([x + t, y + t, x2 - t, y2 - t], outline=color, width=1)
        lab = d.get("label", str(cid))
        txt = f"{lab} {d['score']:.2f}"
        th = getattr(font, "size", 16)
        tw = draw.textlength(txt, font=font)
        draw.rectangle([x, y - th - 2, x + tw + 6, y], fill=color)
        draw.text((x + 3, y - th - 1), txt, fill=(0, 0, 0), font=font)

    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    im.save(vis_path, quality=95)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="yolo11x.pt / yolov8x.pt 等")
    ap.add_argument("--val-img", required=True, help="COCO val2017 图像目录")
    ap.add_argument("--val-ann", required=True, help="instances_val2017.json")
    ap.add_argument("--out", required=True, help="导出 COCO detections JSON 路径")

    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=4)         # 大显存可改 8
    ap.add_argument("--workers", type=int, default=2)       # 过大易触发 open files
    ap.add_argument("--tta", action="store_true")           # 作为 Ultralytics 的 augment
    ap.add_argument("--agnostic", action="store_true", help="二次NMS跨类（更强去重）")
    ap.add_argument("--extra-nms", type=float, default=0.7, help="二次NMS阈值")
    ap.add_argument("--max-det", type=int, default=300)

    ap.add_argument("--vis-dir", default=None)
    ap.add_argument("--vis-conf", type=float, default=0.35)
    args = ap.parse_args()

    # 只读取文件名，不打开文件（避免句柄激增）
    img_fns = sorted([f for f in os.listdir(args.val_img)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"[Info] images: {len(img_fns)}")

    fname2id, name2id = _load_coco_index(args.val_ann)
    model = YOLO(args.model)

    # 流式预测：source=目录，stream=True
    res_iter = model.predict(
        source=args.val_img,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=False,
        verbose=False,
        stream=True,                 # 逐个生成结果
        augment=args.tta,            # TTA
        agnostic_nms=args.agnostic,  # 内部NMS是否跨类
        max_det=args.max_det,
        batch=args.batch,
        workers=args.workers,
    )

    results_total = []
    names_ref = None  # YOLO 类别名表

    for i, r in enumerate(res_iter, 1):
        fn = os.path.basename(r.path)
        if names_ref is None:
            names_ref = r.names  # {id: name}

        # 若该文件不在标注（COCO评测只认标注内的 file_name）
        if fn not in fname2id:
            continue
        image_id = fname2id[fn]

        dets_img = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)

            # 先拿到模型输出（已做过一次 NMS）
            tmp = []
            for j in range(xyxy.shape[0]):
                name = names_ref.get(int(cls[j]), "").strip().lower()
                name = _ALIASES.get(name, name)
                if name not in name2id:
                    continue
                tmp.append({
                    "xyxy": xyxy[j],
                    "score": float(conf[j]),
                    "label": name,
                    "category_id": int(name2id[name]),
                })

            # 二次NMS：类内或跨类
            if tmp:
                boxes = np.stack([d["xyxy"] for d in tmp], axis=0)
                scores = np.array([d["score"] for d in tmp], dtype=np.float32)
                if args.agnostic:
                    keep_idx = _nms_xyxy(boxes, scores, args.extra_nms)
                    tmp = [tmp[k] for k in keep_idx]
                else:
                    kept_all = []
                    by_cls = {}
                    for idx_, d in enumerate(tmp):
                        by_cls.setdefault(d["category_id"], []).append(idx_)
                    for cid, idxs in by_cls.items():
                        b = boxes[idxs]
                        s = scores[idxs]
                        kk = _nms_xyxy(b, s, args.extra_nms)
                        kept_all.extend([idxs[k] for k in kk])
                    kept_all = sorted(set(kept_all), key=lambda k: scores[k], reverse=True)
                    tmp = [tmp[k] for k in kept_all]

            # 转 COCO xywh
            for d in tmp:
                x1, y1, x2, y2 = d["xyxy"]
                x = float(x1); y = float(y1)
                w = float(x2 - x1); h = float(y2 - y1)
                dets_img.append({
                    "image_id": int(image_id),
                    "category_id": int(d["category_id"]),
                    "bbox": [x, y, w, h],
                    "score": float(d["score"]),
                    "label": d["label"],
                })

        # 累加
        results_total.extend(dets_img)

        # 可视化（仅该图的 dets）
        if args.vis_dir is not None:
            vis_path = os.path.join(args.vis_dir, fn)
            img_path = os.path.join(args.val_img, fn)
            _draw_boxes(img_path, dets_img, vis_path, args.vis_conf)

        if (i <= 5) or (i % 100 == 0):
            print(f"[{i}/{len(img_fns)}] {fn}: dets={len(dets_img)} total={len(results_total)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results_total, f)
    print(f"[Save] COCO detections -> {args.out} (items={len(results_total)})")
    print("[Done] 推理完成，可直接用 COCOeval 评估。")


if __name__ == "__main__":
    main()
