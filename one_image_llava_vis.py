#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse, random

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ---------------- Robust parsing ----------------

def _strip_fences(t):
    t = (t or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t).strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    return t

def _loads(s):
    try:
        return json.loads(s)
    except Exception:
        return None

def parse_json_block(text):
    t = _strip_fences(text)
    obj = _loads(t)
    base = t
    if obj is None:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", t)
        if m:
            base = m.group(1)
            obj = _loads(base)
        if obj is None:
            for suf in ("}]}", "}}", "]}", "}"):
                obj = _loads(base + suf)
                if obj is not None:
                    break
    return obj

def parse_dets_and_selfacc(text):
    obj = parse_json_block(text)
    dets, acc = [], None
    if isinstance(obj, dict):
        dets = obj.get("detections") or obj.get("objects") or []
        sr = obj.get("self_report")
        if isinstance(sr, dict) and "accuracy" in sr:
            try:
                v = float(sr["accuracy"])
                acc = v / 100.0 if v > 1.0 else v
            except Exception:
                pass
    elif isinstance(obj, list):
        dets = obj
    if acc is None:
        t = _strip_fences(text)
        m = re.search(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*%', t)
        if m:
            v = float(m.group(1))
            if 0 <= v <= 100:
                acc = v / 100.0
        else:
            m = re.search(r'(?<![0-9])((?:0?\.[0-9]+)|1(?:\.0+)?)', t)
            if m:
                try:
                    acc = max(0.0, min(1.0, float(m.group(1))))
                except Exception:
                    acc = None
    return (dets if isinstance(dets, list) else []), acc

# ---------------- Geometry / NMS ----------------

def iou_xywh(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    iw = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    ih = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter + 1e-6
    return inter / union

def nms_xywh(dets, thr):
    s = sorted(dets, key=lambda d: float(d.get("score", d.get("confidence", 0.5))), reverse=True)
    keep = []
    for d in s:
        ok = True
        for k in keep:
            if iou_xywh(d["bbox"], k["bbox"]) > thr:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep

# ---------------- Chat generation ----------------

PROMPT_BASE = (
    "Detect objects in the image and return ONLY one JSON object with key \"detections\".\n"
    "Each item: {{\"label\":\"<category>\", \"box\":[x1,y1,x2,y2], \"confidence\":0.xx}}.\n"
    "Coordinates normalized to [0,1] with top-left (x1,y1), bottom-right (x2,y2), x1<x2, y1<y2.\n"
    "Round to 3 decimals. At most {max_objects} objects. No markdown/code fences, no explanations."
)

PROMPT_SELFACC = (
    "Detect objects as above, and ALSO include \"self_report\": {{\"accuracy\": A}} where A is YOUR estimated "
    "overall detection accuracy for this image (0~1 or percentage). Output a single JSON object only."
)

def build_prompt(max_objects, want_self_acc):
    p = PROMPT_BASE.format(max_objects=max_objects)
    if want_self_acc:
        p += "\n" + PROMPT_SELFACC
    return p

def generate_text(model, processor, image, prompt, device, max_new_tokens):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=chat, images=image, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
            pad_token_id=(getattr(processor.tokenizer, "pad_token_id", None)
                          or getattr(processor.tokenizer, "eos_token_id", None)),
        )
    seq = out.sequences[0]
    gen_ids = seq[inputs["input_ids"].shape[-1]:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()

# ---------------- Draw ----------------

def _pick_color(name):
    random.seed(hash(name) & 0xffffffff)
    return (random.randint(30, 230), random.randint(30, 230), random.randint(30, 230))

def _text_size(draw, font, txt):
    try:
        w = draw.textlength(txt, font=font)
        h = font.size if hasattr(font, "size") else 16
        return w, h
    except Exception:
        try:
            return font.getsize(txt)
        except Exception:
            return (8 * len(txt), 16)

def draw_boxes(im, dets, lw=3):
    im = im.copy()
    draw = ImageDraw.Draw(im)
    W, H = im.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for d in dets:
        lab = str(d.get("label") or d.get("category") or "obj")
        s = float(d.get("score", d.get("confidence", 0.5)))
        x, y, w, h = d["bbox"]
        c = _pick_color(lab)
        for t in range(lw):
            draw.rectangle([x + t, y + t, x + w - t, y + h - t], outline=c, width=1)
        txt = "%s %.2f" % (lab, s)
        tw, th = _text_size(draw, font, txt)
        draw.rectangle([x, y - th - 2, x + tw + 6, y], fill=c)
        draw.text((x + 3, y - th - 1), txt, fill=(0, 0, 0), font=font)
    return im

# ---------------- Optional per-image metric @0.5 ----------------

def per_image_prf(gt_boxes, pred, iou_thr=0.5):
    matched = set()
    tp = 0
    fp = 0
    for d in sorted(pred, key=lambda x: float(x.get("score", x.get("confidence", 0.5))), reverse=True):
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gt_boxes):
            if j in matched:
                continue
            iou = iou_xywh(d["bbox"], g)
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            matched.add(best_j)
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - tp
    prec = tp / float(max(1, tp + fp))
    rec = tp / float(max(1, tp + fn))
    f1 = 2 * prec * rec / float(max(1e-6, (prec + rec)))
    return tp, fp, fn, prec, rec, f1

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--proc-dir", default=None, help="若合并模型目录缺tokenizer，可指向基座目录")
    ap.add_argument("--image", required=True, help="单张图片路径")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--max-objects", type=int, default=20)
    ap.add_argument("--min-conf", type=float, default=0.00)
    ap.add_argument("--nms-iou", type=float, default=0.60)
    ap.add_argument("--device-map", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--self-report", action="store_true")
    ap.add_argument("--out-img", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--save-raw", action="store_true")
    ap.add_argument("--coco-ann", default=None)
    ap.add_argument("--image-id", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("[Load] %s" % args.model_dir)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map=("cuda:0" if (device == "cuda" and args.device_map == "cuda")
                    else ("auto" if device == "cuda" else None))
    )
    proc_dir = args.proc_dir or args.model_dir
    processor = AutoProcessor.from_pretrained(proc_dir)

    try:
        gc = model.generation_config
        gc.do_sample = False
        gc.num_beams = 1
        for k in ("temperature", "top_p", "top_k", "typical_p"):
            if hasattr(gc, k):
                setattr(gc, k, None)
    except Exception:
        pass

    image = Image.open(args.image).convert("RGB")
    W, H = image.size

    prompt = build_prompt(args.max_objects, args.self_report)
    print("[Gen] start …")
    text = generate_text(model, processor, image, prompt, device, args.tokens)
    print("[Gen] raw text (head):\n%s" % (text[:400] + ("..." if len(text) > 400 else "")))

    dets_raw, self_acc = parse_dets_and_selfacc(text)

    parsed = []
    for d in dets_raw[: args.max_objects * 3]:
        if not isinstance(d, dict):
            continue
        lab = str(d.get("label") or d.get("category") or "").strip()
        box = d.get("box") or d.get("bbox")
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        try:
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        except Exception:
            continue
        normalized = max(x1, y1, x2, y2) <= 1.2
        if normalized:
            x1 = max(0, min(1, x1)); y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2)); y2 = max(0, min(1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            x, y, w, h = x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H
        else:
            if x2 <= x1 or y2 <= y1:
                continue
            x, y, w, h = x1, y1, (x2 - x1), (y2 - y1)
        if w <= 1 or h <= 1:
            continue
        s = d.get("confidence", d.get("score", 0.5))
        try:
            s = float(s)
        except Exception:
            s = 0.5
        s = max(0.0, min(1.0, s))
        parsed.append({"label": lab, "bbox": [x, y, w, h], "score": s})

    kept = [r for r in parsed if r["score"] >= args.min_conf]
    bycls = {}
    for r in kept:
        bycls.setdefault(r["label"], []).append(r)
    kept2 = []
    for _, items in bycls.items():
        kept2.extend(nms_xywh(items, args.nms_iou))
    kept3 = nms_xywh(kept2, args.nms_iou)
    kept3 = sorted(kept3, key=lambda d: d["score"], reverse=True)[: args.max_objects]

    print("[Info] kept %d boxes; self_report accuracy = %s" % (len(kept3), str(self_acc)))

    out_img = args.out_img or (os.path.splitext(args.image)[0] + "_llava_det.jpg")
    vis = draw_boxes(image, kept3, lw=3)
    vis.save(out_img, quality=95)
    print("[Save] vis -> %s" % out_img)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(kept3, f, ensure_ascii=False, indent=2)
        print("[Save] json -> %s" % args.out_json)

    if args.save_raw:
        raw_path = os.path.splitext(out_img)[0] + ".raw.txt"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("[Save] raw -> %s" % raw_path)

    if args.coco_ann and args.image_id is not None:
        try:
            from pycocotools.coco import COCO
            coco = COCO(args.coco_ann)
            ganns = coco.loadAnns(coco.getAnnIds(imgIds=[args.image_id]))
            gtb = [[a["bbox"][0], a["bbox"][1], a["bbox"][2], a["bbox"][3]] for a in ganns]
            tp, fp, fn, prec, rec, f1 = per_image_prf(gtb, kept3, iou_thr=0.5)
            print("[COCO@0.5 single-image] TP=%d FP=%d FN=%d  Precision=%.3f Recall=%.3f F1=%.3f" %
                  (tp, fp, fn, prec, rec, f1))
        except Exception as e:
            print("[Warn] single-image metric failed: %s" % e)

if __name__ == "__main__":
    main()
