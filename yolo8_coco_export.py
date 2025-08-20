#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 Ultralytics YOLOv8/YOLO11 对 COCO val2017 推理：
- 导出 COCO 格式 detections JSON：[{image_id, category_id, bbox[x,y,w,h], score}, ...]
- 可选保存可视化图（画框）
- 支持 --limit N 只跑前 N 张（或 --shuffle 随机取 N 张）
"""

import os, json, argparse, random
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def load_coco_index(ann_path: str):
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    fname2id = {im["file_name"]: int(im["id"]) for im in ann["images"]}
    name2id = {c["name"].strip().lower(): int(c["id"]) for c in ann["categories"]}
    return fname2id, name2id

def xyxy_to_xywh(x1, y1, x2, y2):
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)

def draw_boxes_pil(img_path: str, dets: List[dict], vis_path: str):
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for d in dets:
        x, y, w, h = d["bbox"]; x2, y2 = x+w, y+h
        cid = d.get("category_id", 0)
        color = (50 + (cid*37)%180, 50 + (cid*81)%180, 50 + (cid*113)%180)
        for t in range(3):
            draw.rectangle([x+t, y+t, x2-t, y2-t], outline=color, width=1)
        lab = str(d.get("label", "")) or str(d.get("category_id", ""))
        txt = f"{lab} {d['score']:.2f}"
        tw = draw.textlength(txt, font=font)
        th = getattr(font, "size", 16)
        draw.rectangle([x, y-th-2, x+tw+6, y], fill=color)
        draw.text((x+3, y-th-1), txt, fill=(0,0,0), font=font)
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    im.save(vis_path, quality=95)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="yolov8s.pt / yolo11s.pt 等")
    ap.add_argument("--val-img", required=True, help="例如 /home/vipuser/coco/images/val2017")
    ap.add_argument("--val-ann", required=True, help="例如 /home/vipuser/coco/annotations/instances_val2017.json")
    ap.add_argument("--out", required=True, help="导出的 COCO detections JSON 路径")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.7)
    ap.add_argument("--device", default="0")  # "0" / "cpu"
    ap.add_argument("--vis-dir", default=None, help="若设定则保存可视化图片到该目录")
    ap.add_argument("--limit", type=int, default=50, help="仅处理前 N 张（或随机 N 张，配合 --shuffle）")
    ap.add_argument("--shuffle", action="store_true", help="随机抽取 N 张（默认按文件名排序取前 N）")
    args = ap.parse_args()

    fname2id, name2id = load_coco_index(args.val_ann)
    model = YOLO(args.model)

    img_files = sorted([f for f in os.listdir(args.val_img)
                        if f.lower().endswith((".jpg",".jpeg",".png"))])
    if args.shuffle:
        random.seed(0)
        random.shuffle(img_files)
    if args.limit and args.limit > 0:
        img_files = img_files[: args.limit]

    results_total = []
    for i, fn in enumerate(img_files, 1):
        img_path = os.path.join(args.val_img, fn)
        image_id = fname2id.get(fn)
        if image_id is None:
            print(f"[Warn] {fn} 不在标注文件中，跳过"); continue

        res = model.predict(
            source=img_path, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            device=args.device, save=False, verbose=False
        )
        r = res[0]
        dets_img = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names  # {id: name}

            for j in range(xyxy.shape[0]):
                x1,y1,x2,y2 = xyxy[j]
                score = float(conf[j])
                cls_id = int(cls[j])
                name = names.get(cls_id, "").strip().lower()
                # YOLOv8 的 COCO 名称基本与 COCO 一致；个别名称做兜底
                alias = {
                    "dining table":"dining table", "potted plant":"potted plant",
                    "hair drier":"hair drier", "cell phone":"cell phone",
                    "wine glass":"wine glass", "sports ball":"sports ball"
                }
                name = alias.get(name, name)
                if name not in name2id:
                    # 未映射的极少数类名，跳过以免评测出错
                    # print(f"[Note] Unmapped class: {names.get(cls_id, cls_id)}")
                    continue
                cat_id = name2id[name]
                x,y,w,h = xyxy_to_xywh(x1,y1,x2,y2)
                det = {
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x, y, w, h],
                    "score": score,
                    "label": name  # 仅可视化用
                }
                results_total.append(det)
                dets_img.append(det)

        if args.vis_dir:
            vis_path = os.path.join(args.vis_dir, fn)
            try:
                draw_boxes_pil(img_path, dets_img, vis_path)
            except Exception as e:
                print(f"[Warn] 可视化失败 {fn}: {e}")

        if (i <= 5) or (i % 10 == 0):
            print(f"[{i}/{len(img_files)}] {fn}: dets={len(dets_img)}  total={len(results_total)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results_total, f)
    print(f"[Save] COCO detections -> {args.out}  (items={len(results_total)})")

if __name__ == "__main__":
    main()
