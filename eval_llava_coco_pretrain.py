#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, json, math, os, re, time, random
from pathlib import Path
from typing import List, Dict, Any

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ---------- COCO 80 类（字符串 -> 官方 category_id） ----------
COCO80 = {
    "person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5,
    "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10,
    "fire hydrant": 11, "stop sign": 13, "parking meter": 14, "bench": 15,
    "bird": 16, "cat": 17, "dog": 18, "horse": 19, "sheep": 20, "cow": 21,
    "elephant": 22, "bear": 23, "zebra": 24, "giraffe": 25,
    "backpack": 27, "umbrella": 28, "handbag": 31, "tie": 32, "suitcase": 33,
    "frisbee": 34, "skis": 35, "snowboard": 36, "sports ball": 37, "kite": 38,
    "baseball bat": 39, "baseball glove": 40, "skateboard": 41, "surfboard": 42, "tennis racket": 43,
    "bottle": 44, "wine glass": 46, "cup": 47, "fork": 48, "knife": 49, "spoon": 50, "bowl": 51,
    "banana": 52, "apple": 53, "sandwich": 54, "orange": 55, "broccoli": 56, "carrot": 57,
    "hot dog": 58, "pizza": 59, "donut": 60, "cake": 61, "chair": 62, "couch": 63,
    "potted plant": 64, "bed": 65, "dining table": 67, "toilet": 70, "tv": 72, "laptop": 73,
    "mouse": 74, "remote": 75, "keyboard": 76, "cell phone": 77, "microwave": 78, "oven": 79,
    "toaster": 80, "sink": 81, "refrigerator": 82, "book": 84, "clock": 85, "vase": 86,
    "scissors": 87, "teddy bear": 88, "hair drier": 89, "toothbrush": 90
}
ALLOWED_SET = set(COCO80.keys())


# ---------- 提示词：干净、无类名长表、带约束 ----------
INSTRUCTION = (
    "You are an object detection assistant. "
    "Return ONLY a valid JSON with key 'detections'. "
    "Each item: {\"label\": <one COCO class>, \"box\": [x1,y1,x2,y2], \"confidence\": [0,1]}. "
    "Coordinates are normalized to [0,1] with (x1,y1)=top-left and (x2,y2)=bottom-right. "
    "At most 6 objects. Do NOT repeat the same box or class. "
    "If two boxes IoU>0.6, keep the one with higher confidence. "
    "If nothing is found, return {\"detections\":[]}. Output JSON only."
)


# ---------- 工具函数 ----------
def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    areaA = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    areaB = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = areaA + areaB - inter + 1e-9
    return inter / union


def robust_json_slice(text: str) -> str:
    """在长文本里切出最外层 { ... } """
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        return text[s:e + 1]
    return ""


def parse_and_postprocess(raw_text: str) -> List[Dict[str, Any]]:
    """
    解析大模型输出，过滤非法项并做 NMS/去重/限数
    返回标准化后的 detections 列表
    """
    body = robust_json_slice(raw_text)
    if not body:
        return None
    try:
        data = json.loads(body)
    except Exception:
        return None

    dets = data.get("detections", [])
    clean = []
    for d in dets:
        try:
            label = str(d.get("label", "")).strip().lower()
            if label not in ALLOWED_SET:
                continue
            box = [float(x) for x in d.get("box", [])]
            if len(box) != 4:
                continue
            x1 = max(0.0, min(1.0, box[0]))
            y1 = max(0.0, min(1.0, box[1]))
            x2 = max(0.0, min(1.0, box[2]))
            y2 = max(0.0, min(1.0, box[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            conf = float(d.get("confidence", 0.0))
            # 量化，减少“几乎相同但不同 token”的重复
            bx = [round(x, 3) for x in [x1, y1, x2, y2]]
            clean.append({"label": label, "box": bx, "confidence": conf})
        except Exception:
            continue

    # 置信度排序
    clean.sort(key=lambda z: z["confidence"], reverse=True)

    # 同类 NMS + 限制最多 K 个
    kept = []
    K = 6
    thr = 0.6
    for d in clean:
        dup = False
        for k in kept:
            if d["label"] == k["label"] and iou(d["box"], k["box"]) > thr:
                dup = True
                break
        if not dup:
            kept.append(d)
            if len(kept) >= K:
                break
    return kept


def boxes_to_coco(dets: List[Dict[str, Any]], image_id: int, W: int, H: int) -> List[Dict[str, Any]]:
    """将归一化 [x1,y1,x2,y2] 转换为 COCO 的 [x,y,w,h] 绝对像素，并映射到 category_id"""
    out = []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        x = max(0, min(W - 1, int(round(x1 * W))))
        y = max(0, min(H - 1, int(round(y1 * H))))
        x2p = max(0, min(W - 1, int(round(x2 * W))))
        y2p = max(0, min(H - 1, int(round(y2 * H))))
        w = max(0, x2p - x)
        h = max(0, y2p - y)
        if w <= 0 or h <= 0:
            continue
        cat_id = COCO80[d["label"]]
        out.append({
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [float(x), float(y), float(w), float(h)],
            "score": float(d["confidence"])
        })
    return out


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--proc-dir",  type=str, default=None, help="若合并目录缺少处理器配置，可指定基座目录")
    parser.add_argument("--val-ann",   type=str, required=True)
    parser.add_argument("--val-img",   type=str, required=True)
    parser.add_argument("--subset",    type=int, default=500)
    parser.add_argument("--tokens",    type=int, default=768)
    parser.add_argument("--out",       type=str, required=True)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--show-raw",  action="store_true", help="打印前几张原始回复")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    # 加载模型 & 处理器（processor 可与模型分离）
    print(f"[info] loading model from: {args.model-dir}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=None
    ).to(device)
    proc_dir = args.proc_dir if args.proc_dir else args.model_dir
    print(f"[info] loading processor from: {proc_dir}")
    processor = AutoProcessor.from_pretrained(proc_dir)

    # 生成参数：抑制复读 + 充足长度
    gen_kwargs = dict(
        do_sample=False,
        max_new_tokens=int(args.tokens),
        repetition_penalty=1.25,
        no_repeat_ngram_size=12,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=model.generation_config.eos_token_id,
    )

    # COCO 数据
    coco = COCO(args.val_ann)
    img_root = Path(args.val_img)
    img_ids = coco.getImgIds()
    img_ids = img_ids[:args.subset] if args.subset > 0 else img_ids
    print("index created!")
    print(f"[info] subset = {len(img_ids)} images")

    detections_all = []
    ok_json = 0
    t0 = time.time()

    for i, img_id in enumerate(img_ids, 1):
        info = coco.loadImgs([img_id])[0]
        img_path = img_root / info["file_name"]
        W, H = int(info["width"]), int(info["height"])

        image = Image.open(img_path).convert("RGB")

        # 构建多模态 chat messages（LLaVA-HF 推荐写法）
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION}
            ]
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.inference_mode():
            out_ids = model.generate(**inputs, **gen_kwargs)
        text = processor.decode(out_ids[0], skip_special_tokens=True)

        if args.show_raw and i <= 5:
            print(f"==== RAW REPLY (image {i}: {info['file_name']}) ====")
            print(text)

        parsed = parse_and_postprocess(text)
        if parsed is not None:
            ok_json += 1
            dets_coco = boxes_to_coco(parsed, img_id, W, H)
            detections_all.extend(dets_coco)
            dt = len(dets_coco)
        else:
            dt = 0

        print(f"[{i}/{len(img_ids)}] dt={dt}")

    # 保存预测
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(detections_all, f)
    print(f"[saved] detections -> {out_path}")

    # JSON 可解析率
    rate = ok_json / max(1, len(img_ids)) * 100.0
    print(f"[info] JSON success rate: {ok_json}/{len(img_ids)} = {rate:.2f}%")

    print(f"[time] processed {len(img_ids)} images in {time.time() - t0:.1f}s")

    # mAP/AR
    if len(detections_all) > 0:
        coco_dt = coco.loadRes(str(out_path))
        evaluator = COCOeval(coco, coco_dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        # 方便 grep 的一行摘要
        stats = evaluator.stats  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
        print(f"[metrics] mAP@[.50:.95]={stats[0]:.4f} | AP50={stats[1]:.4f} | AR@100={stats[8]:.4f}")
    else:
        print("[warn] empty detections; skip COCOeval.")


if __name__ == "__main__":
    main()
