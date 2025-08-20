#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, random
from typing import Dict, List
from PIL import Image, ImageDraw, ImageFont

def load_coco_maps(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    id2name = {int(c["id"]): c["name"] for c in ann["categories"]}
    id2file = {int(im["id"]): im["file_name"] for im in ann["images"]}
    return id2name, id2file

def draw_one(img_path: str, dets: List[dict], out_path: str, vis_conf: float):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with Image.open(img_path) as _im:
        im = _im.convert("RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for d in dets:
        if float(d.get("score", 0.0)) < vis_conf: 
            continue
        x, y, w, h = d["bbox"]
        x2, y2 = x + w, y + h
        cid = int(d.get("category_id", 0))
        color = (50 + (cid * 37) % 180, 50 + (cid * 81) % 180, 50 + (cid * 113) % 180)
        for t in range(3):
            draw.rectangle([x + t, y + t, x2 - t, y2 - t], outline=color, width=1)
        lab = d.get("label", str(cid))
        txt = f"{lab} {float(d.get('score',0.0)):.2f}"
        th = getattr(font, "size", 16)
        tw = draw.textlength(txt, font=font)
        draw.rectangle([x, y - th - 2, x + tw + 6, y], fill=color)
        draw.text((x + 3, y - th - 1), txt, fill=(0, 0, 0), font=font)

    im.save(out_path, quality=95)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-img", required=True, help="COCO val2017 图像目录")
    ap.add_argument("--val-ann", required=True, help="instances_val2017.json")
    ap.add_argument("--dets",    required=True, help="导出的 detections JSON 路径")
    ap.add_argument("--vis-dir", required=True, help="输出可视化目录")
    ap.add_argument("--vis-conf", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=20, help="最多可视化多少张（默认20）")
    ap.add_argument("--filenames", type=str, default="", help="逗号分隔的文件名列表，仅渲染这些（可选）")
    ap.add_argument("--ids", type=str, default="", help="逗号分隔的image_id列表，仅渲染这些（可选）")
    args = ap.parse_args()

    id2name, id2file = load_coco_maps(args.val_ann)

    # 读入 detections，按 image_id 归组
    with open(args.dets, "r", encoding="utf-8") as f:
        dets = json.load(f)
    by_img: Dict[int, List[dict]] = {}
    for d in dets:
        by_img.setdefault(int(d["image_id"]), []).append(d)

    # 选择要渲染的 image_id 集合
    targets = []
    if args.filenames:
        want = set(s.strip() for s in args.filenames.split(",") if s.strip())
        targets = [img_id for img_id, fn in id2file.items() if fn in want]
    elif args.ids:
        targets = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        # 随机采样 limit 张（只采有检测结果的）
        pool = list(by_img.keys())
        random.shuffle(pool)
        targets = pool[: args.limit]

    os.makedirs(args.vis_dir, exist_ok=True)
    print(f"[Info] Will render {len(targets)} images to: {args.vis_dir}")

    for k, img_id in enumerate(targets, 1):
        fn = id2file.get(img_id)
        if not fn: 
            continue
        img_path = os.path.join(args.val_img, fn)
        vis_path = os.path.join(args.vis_dir, fn)
        # 填充 label 文本
        dets_img = []
        for d in by_img.get(img_id, []):
            lab = d.get("label")
            if not lab:
                lab = id2name.get(int(d["category_id"]), str(d["category_id"]))
            dd = dict(d)
            dd["label"] = lab
            dets_img.append(dd)
        draw_one(img_path, dets_img, vis_path, args.vis_conf)
        if (k <= 5) or (k % 50 == 0):
            print(f"[{k}/{len(targets)}] {fn} dets={len(dets_img)}")

if __name__ == "__main__":
    main()
