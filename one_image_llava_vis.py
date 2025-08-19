#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse, math, random
from typing import Any, Dict, List, Tuple, Optional

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ---------------- Robust parsing ----------------

def _strip_fences(t: str) -> str:
    t = t.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t).strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    return t

def _loads(s: str):
    try: return json.loads(s)
    except Exception: return None

def parse_json_block(text: str) -> Optional[dict | list]:
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

def parse_dets_and_selfacc(text: str) -> Tuple[List[Dict[str,Any]], Optional[float]]:
    obj = parse_json_block(text)
    dets, acc = [], None
    if isinstance(obj, dict):
        dets = obj.get("detections") or obj.get("objects") or []
        sr = obj.get("self_report")
        if isinstance(sr, dict) and "accuracy" in sr:
            try:
                v = float(sr["accuracy"])
                acc = v/100.0 if v > 1.0 else v
            except: pass
    elif isinstance(obj, list):
        dets = obj
    # 兜底：百分比或小数
    if acc is None:
        t = _strip_fences(text)
        m = re.search(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*%', t)
        if m:
            v = float(m.group(1)); 
            if 0 <= v <= 100: acc = v/100.0
        else:
            m = re.search(r'(?<![0-9])((?:0?\.[0-9]+)|1(?:\.0+)?)', t)
            if m:
                acc = max(0.0, min(1.0, float(m.group(1))))
    return (dets if isinstance(dets,list) else []), acc

# ---------------- Geometry / NMS ----------------

def iou_xywh(b1, b2) -> float:
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    xa1,ya1,xa2,ya2 = x1, y1, x1+w1, y1+h1
    xb1,yb1,xb2,yb2 = x2, y2, x2+w2, y2+h2
    iw = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    ih = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = iw*ih
    union = w1*h1 + w2*h2 - inter + 1e-6
    return inter / union

def nms_xywh(dets: List[Dict[str,Any]], thr: float) -> List[Dict[str,Any]]:
    s = sorted(dets, key=lambda d: float(d.get("score", d.get("confidence", 0.5))), reverse=True)
    keep=[]
    for d in s:
        ok=True
        for k in keep:
            if iou_xywh(d["bbox"], k["bbox"]) > thr:
                ok=False; break
        if ok: keep.append(d)
    return keep

# ---------------- Chat generation ----------------

PROMPT_BASE = (
    "Detect objects in the image and return ONLY one JSON object with key \"detections\".\n"
    "Each item: {{\"label\":\"<category>\", \"box\":[x1,y1,x2,y2], \"confidence\":0.xx}}.\n"
    "Coordinates normalized to [0,1] with top-left (x1,y1), bottom-right (x2,y2), x1<x2, y1<y2.\n"
    "Round to 3 decimals. At most {max_objects} objects. No markdown/code fences, no explanations."
)

PROMPT_SELFACC = (
    "Detect objects as above, and ALSO include \"self_report\": {{\"accuracy\": A}} where A is YOUR estimated"
    " overall detection accuracy for this image (0~1 or percentage). Output a single JSON object only."
)

def build_prompt(max_objects: int, want_self_acc: bool) -> str:
    p = PROMPT_BASE.format(max_objects=max_objects)
    if want_self_acc: p += "\n" + PROMPT_SELFACC
    return p

def generate_text(model, processor, image: Image.Image, prompt: str, device: str, max_new_tokens: int) -> str:
    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=chat, images=image, return_tensors="pt")
    inputs = {k:(v.to(device) if hasattr(v,"to") else v) for k,v in inputs.items()}
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

def color_for(name: str) -> Tuple[int,int,int]:
    random.seed(hash(name) & 0xffffffff)
    return tuple(int(x) for x in (random.randint(30,230), random.randint(30,230), random.randint(30,230)))

def draw_boxes(im: Image.Image, dets: List[Dict[str,Any]], lw: int=3) -> Image.Image:
    im = im.copy()
    draw = ImageDraw.Draw(im)
    W,H = im.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    for d in dets:
        lab = str(d.get("label") or d.get("category") or "obj")
        s = float(d.get("score", d.get("confidence", 0.5)))
        x,y,w,h = d["bbox"]
        c = color_for(lab)
        # rectangle
        for t in range(lw):
            draw.rectangle([x+t,y+t,x+w-t,y+h-t], outline=c, width=1)
        # text
        txt = f"{lab} {s:.2f}"
        tw,th = draw.textlength(txt, font=font), 16
        draw.rectangle([x, y-th-2, x+tw+6, y], fill=c)
        draw.text((x+3, y-th-1), txt, fill=(0,0,0), font=font)
    return im

# ---------------- Single-image "real" @IoU=0.5 (optional) ----------------

def per_image_prf(gt_boxes: List[List[float]], pred: List[Dict[str,Any]], iou_thr=0.5) -> Tuple[int,int,int,float,float,float]:
    matched = set()
    tp=0; fp=0
    for d in sorted(pred, key=lambda x: float(x.get("score", x.get("confidence",0.5))), reverse=True):
        best_iou=0.0; best_j=-1
        for j, g in enumerate(gt_boxes):
            if j in matched: continue
            if iou_xywh(d["bbox"], g) >= iou_thr and iou_xywh(d["bbox"], g) > best_iou:
                best_iou = iou_xywh(d["bbox"], g); best_j=j
        if best_j>=0:
            matched.add(best_j); tp+=1
        else:
            fp+=1
    fn = len(gt_boxes)-tp
    prec = tp/max(1,tp+fp)
    rec  = tp/max(1,tp+fn)
    f1   = 2*prec*rec/max(1e-6, (prec+rec))
    return tp,fp,fn,prec,rec,f1

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--proc-dir", default=None, help="可选：处理器从这里加载（若合并目录缺tokenizer）")
    ap.add_argument("--image", required=True, help="单张图片路径")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--max-objects", type=int, default=20)
    ap.add_argument("--min-conf", type=float, default=0.00)
    ap.add_argument("--nms-iou", type=float, default=0.60)
    ap.add_argument("--device-map", choices=["auto","cuda","cpu"], default="auto")
    ap.add_argument("--self-report", action="store_true", help="要求模型输出 self_report.accuracy")
    ap.add_argument("--out-img", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--save-raw", action="store_true")
    # optional real metric for this image (need COCO-like xywh pixel boxes)
    ap.add_argument("--coco-ann", default=None, help="instances_val2017.json")
    ap.add_argument("--image-id", type=int, default=None, help="该图在标注里的 image_id")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device=="cuda" else torch.float32

    print(f"[Load] {args.model-dir}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map=("cuda:0" if (device=="cuda" and args.device_map=="cuda") else
                    ("auto" if device=="cuda" else None))
    )
    proc_dir = args.proc_dir or args.model_dir
    processor = AutoProcessor.from_pretrained(proc_dir)

    # tidy generation warnings
    try:
        gc = model.generation_config
        gc.do_sample=False; gc.num_beams=1
        for k in ("temperature","top_p","top_k","typical_p"):
            if hasattr(gc,k): setattr(gc,k, None)
    except: pass

    image = Image.open(args.image).convert("RGB")
    W,H = image.size

    prompt = build_prompt(args.max_objects, args.self_report)
    print("[Gen] start …")
    text = generate_text(model, processor, image, prompt, device, args.tokens)
    print("[Gen] raw text:\n", (text[:400] + ("..." if len(text)>400 else "")))

    dets_raw, self_acc = parse_dets_and_selfacc(text)

    # collect raw -> normalize to pixel xywh
    parsed=[]
    for d in dets_raw[: args.max_objects*3]:
        if not isinstance(d, dict): continue
        lab = str(d.get("label") or d.get("category") or "").strip()
        box = d.get("box") or d.get("bbox")
        if not (isinstance(box,(list,tuple)) and len(box)==4): continue
        try:
            x1,y1,x2,y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        except: 
            continue
        normalized = max(x1,y1,x2,y2) <= 1.2
        if normalized:
            x1=max(0,min(1,x1)); y1=max(0,min(1,y1))
            x2=max(0,min(1,x2)); y2=max(0,min(1,y2))
            if x2<=x1 or y2<=y1: continue
            x,y,w,h = x1*W, y1*H, (x2-x1)*W, (y2-y1)*H
        else:
            if x2<=x1 or y2<=y1: continue
            x,y,w,h = x1, y1, (x2-x1), (y2-y1)
        if w<=1 or h<=1: continue
        s = d.get("confidence", d.get("score", 0.5))
        try: s = float(s)
        except: s = 0.5
        s = max(0.0, min(1.0, s))
        parsed.append({"label":lab, "bbox":[x,y,w,h], "score":s})

    # post: min-conf -> classwise nms -> global nms -> topK
    kept = [r for r in parsed if r["score"] >= args.min_conf]
    # 类内 NMS
    bycls: Dict[str, List[Dict[str,Any]]] = {}
    for r in kept: bycls.setdefault(r["label"], []).append(r)
    kept2=[]
    for _, items in bycls.items(): kept2.extend(nms_xywh(items, args.nms_iou))
    kept3 = nms_xywh(kept2, args.nms_iou)
    kept3 = sorted(kept3, key=lambda d:d["score"], reverse=True)[: args.max_objects]

    print(f"[Info] kept {len(kept3)} boxes; self_report accuracy = {self_acc}")

    # save image
    out_img = args.out_img or os.path.splitext(args.image)[0] + "_llava_det.jpg"
    vis = draw_boxes(image, kept3, lw=3)
    vis.save(out_img, quality=95)
    print(f"[Save] vis -> {out_img}")

    # save json
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(kept3, f, ensure_ascii=False, indent=2)
        print(f"[Save] json -> {args.out_json}")

    if args.save_raw:
        raw_path = os.path.splitext(out_img)[0] + ".raw.txt"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[Save] raw -> {raw_path}")

    # optional: real per-image PR at IoU=0.5 if COCO ann provided
    if args.coco_ann and args.image_id is not None:
        try:
            from pycocotools.coco import COCO
            coco = COCO(args.coco_ann)
            ganns = coco.loadAnns(coco.getAnnIds(imgIds=[args.image_id]))
            gtb = [[a["bbox"][0], a["bbox"][1], a["bbox"][2], a["bbox"][3]] for a in ganns]
            tp,fp,fn,prec,rec,f1 = per_image_prf(gtb, kept3, iou_thr=0.5)
            print(f"[COCO@0.5 single-image] TP={tp} FP={fp} FN={fn}  Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")
        except Exception as e:
            print(f"[Warn] single-image metric failed: {e}")

if __name__ == "__main__":
    main()
