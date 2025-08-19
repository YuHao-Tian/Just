#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse, time, random
from typing import List, Dict, Any, Tuple, Optional

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ========= Robust JSON parsing =========

def _strip_fences(t: str) -> str:
    t = (t or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t).strip()
        if t.endswith("```"): t = t[:-3].strip()
    return t

def _loads(s: str):
    try: return json.loads(s)
    except Exception: return None

def parse_json(text: str) -> Optional[Dict[str,Any] | List[Any]]:
    t = _strip_fences(text)
    obj = _loads(t); base = t
    if obj is None:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", t)
        if m:
            base = m.group(1); obj = _loads(base)
        if obj is None:
            for suf in ("}]}", "}}", "]}", "}"):
                obj = _loads(base + suf)
                if obj is not None: break
    return obj

def parse_dets_and_selfacc(text: str) -> Tuple[List[Dict[str,Any]], Optional[float]]:
    obj = parse_json(text)
    dets, acc = [], None
    if isinstance(obj, dict):
        dets = obj.get("detections") or obj.get("objects") or []
        sr = obj.get("self_report")
        if isinstance(sr, dict) and "accuracy" in sr:
            try:
                v = float(sr["accuracy"]); acc = v/100.0 if v > 1.0 else v
            except: pass
    elif isinstance(obj, list):
        dets = obj
    if acc is None:
        t = _strip_fences(text)
        m = re.search(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*%', t)
        if m:
            v = float(m.group(1)); acc = v/100.0
        else:
            m = re.search(r'(?<![0-9])((?:0?\.[0-9]+)|1(?:\.0+)?)', t)
            if m:
                try: acc = max(0.0, min(1.0, float(m.group(1))))
                except: acc = None
    return (dets if isinstance(dets, list) else []), acc

# ========= Geometry, IoU / NMS =========

def iou_xywh(b1, b2) -> float:
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    xa1,ya1,xa2,ya2 = x1, y1, x1+w1, y1+h1
    xb1,yb1,xb2,yb2 = x2, y2, x2+w2, y2+h2
    iw = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    ih = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = iw*ih; union = w1*h1 + w2*h2 - inter + 1e-6
    return inter / union

def nms_xywh(dets: List[Dict[str,Any]], thr: float) -> List[Dict[str,Any]]:
    s = sorted(dets, key=lambda d: float(d.get("score", d.get("confidence", 0.5))), reverse=True)
    keep: List[Dict[str,Any]] = []
    for d in s:
        ok=True
        for k in keep:
            if iou_xywh(d["bbox"], k["bbox"]) > thr: ok=False; break
        if ok: keep.append(d)
    return keep

# ========= Label canon =========

COMMON_ALIASES = {
    # 只列出影响大的别名；其余走小写/去复数规则
    "tv": "tv", "television": "tv", "tv monitor": "tv", "tvmonitor": "tv",
    "motorbike": "motorcycle", "aeroplane": "airplane",
    "trafficlight": "traffic light", "firehydrant": "fire hydrant", "hydrant": "fire hydrant",
    "stop sign": "stop sign", "sportsball": "sports ball", "wineglass": "wine glass",
    "tennis-racket": "tennis racket", "baseball-bat": "baseball bat", "baseball-glove": "baseball glove",
    "sofa": "couch", "bike":"bicycle", "handbag":"handbag", "back pack":"backpack",
    "cell phone":"cell phone", "mobile phone":"cell phone",
    "hotdog":"hot dog", "teddybear":"teddy bear", "hairdryer":"hair drier",
    "potted plant":"potted plant", "dining table":"dining table",
    # 口语合称 → 最可能的 COCO 类
    "pots":"bowl", "pans":"bowl", "pots and pans":"bowl", "dishes":"bowl", "plates":"bowl"
}

def canon_label(name: str, valid: set) -> Optional[str]:
    n = (name or "").strip().lower()
    n = n.replace("_"," ").replace("-"," ")
    while "  " in n: n = n.replace("  ", " ")
    n = COMMON_ALIASES.get(n, n)
    if n.endswith("s") and n[:-1] in valid: n = n[:-1]     # 复数→单数
    return n if n in valid else None

# ========= xyxy / xywh 自适应（像素或归一化） =========

def to_xywh_pixels(box, W, H) -> Optional[Tuple[float,float,float,float]]:
    try:
        x1,y1,u3,u4 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    except: return None
    def as_xyxy():
        if max(x1,y1,u3,u4) <= 1.2:   # 规范化
            xx1,yy1,xx2,yy2 = max(0,min(1,x1)), max(0,min(1,y1)), max(0,min(1,u3)), max(0,min(1,u4))
            if xx2<=xx1 or yy2<=yy1: return None
            return xx1*W, yy1*H, (xx2-xx1)*W, (yy2-yy1)*H
        else:                         # 像素
            if u3<=x1 or u4<=y1: return None
            return x1, y1, (u3-x1), (u4-y1)
    def as_xywh():
        if max(x1,y1,u3,u4) <= 1.2:
            return x1*W, y1*H, u3*W, u4*H
        else:
            return x1, y1, u3, u4
    # 先按 xyxy
    r = as_xyxy()
    def area_frac(r): 
        return 1e9 if r is None else max(1.0, r[2]*r[3])/(W*H)
    if r is None or area_frac(r) > 0.9:   # 失败或“过大框”→当作 xywh
        r = as_xywh()
    if r[2]<=1 or r[3]<=1: return None
    return r

# ========= 绘制 =========

def color_for(name: str) -> Tuple[int,int,int]:
    random.seed(hash(name) & 0xffffffff)
    return (random.randint(30,230), random.randint(30,230), random.randint(30,230))

def draw_boxes(im: Image.Image, dets: List[Dict[str,Any]], lw: int=3) -> Image.Image:
    im = im.copy(); draw = ImageDraw.Draw(im)
    try: font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except: font = ImageFont.load_default()
    for d in dets:
        lab = str(d.get("label") or d.get("category") or "obj")
        s = float(d.get("score", d.get("confidence", 0.5)))
        x,y,w,h = d["bbox"]; c = color_for(lab)
        for t in range(lw): draw.rectangle([x+t,y+t,x+w-t,y+h-t], outline=c, width=1)
        txt = f"{lab} {s:.2f}"
        draw.rectangle([x, y-18, x+6+8*len(txt), y], fill=c)
        draw.text((x+3, y-16), txt, fill=(0,0,0), font=font)
    return im

# ========= 生成（chat template 正确喂图） =========

PROMPT_CORE = (
"Detect objects in the image and return ONLY ONE JSON object with key \"detections\".\n"
"Each item must be {{\"label\":\"<one_of_COCO80>\", \"box\":[x1,y1,x2,y2] or [x,y,w,h], \"confidence\":0.xx}}.\n"
"Coordinates can be normalized [0,1] or pixels, but be self-consistent in one image. "
"Boxes must tightly enclose single objects (NO boxes covering the whole image, NO grouping many objects into one box).\n"
"Round to 3 decimals. Use ONLY these labels (singular, English): {cls}.\n"
"At most {maxk} objects. Omit low-confidence objects. Output JSON only, no extra text."
)

PROMPT_SELFACC = (
"Additionally include \"self_report\": {{\"accuracy\": A}} where A is your estimated overall detection accuracy (0~1 or percent)."
)

def build_prompt(classes: List[str], k: int, want_self_acc: bool) -> str:
    p = PROMPT_CORE.format(cls=", ".join(classes), maxk=k)
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
            do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            eos_token_id=getattr(processor.tokenizer,"eos_token_id",None),
            pad_token_id=(getattr(processor.tokenizer,"pad_token_id",None)
                          or getattr(processor.tokenizer,"eos_token_id",None)),
        )
    seq = out.sequences[0]
    gen_ids = seq[inputs["input_ids"].shape[-1]:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()

# ========= pipeline（共享） =========

def postprocess_one(text: str, W: int, H: int, valid_names: Optional[set],
                    name2id: Optional[dict], min_conf: float, nms_iou: float,
                    max_objects: int, drop_giant: bool=True) -> Tuple[List[Dict[str,Any]], Optional[float], List[Dict[str,Any]]]:
    dets_raw, self_acc = parse_dets_and_selfacc(text)

    raw: List[Dict[str,Any]] = []
    for d in dets_raw[: max_objects*4]:
        if not isinstance(d, dict): continue
        lab = str(d.get("label") or d.get("category") or "").strip()
        if valid_names is not None:
            lab = canon_label(lab, valid_names) or ""   # 对不上的直接丢
            if not lab: continue
        box = d.get("box") or d.get("bbox")
        if not (isinstance(box,(list,tuple)) and len(box)==4): continue
        conv = to_xywh_pixels(box, W, H)
        if conv is None: continue
        x,y,w,h = conv
        # 过滤“超大/贴边框”（只用于可视化/单图 sanity；COCOeval 时也可开，建议先开）
        if drop_giant:
            if (w*h)/(W*H) > 0.80: continue
            if x<=2 and y<=2 and x+w>=W-2 and y+h>=H-2: continue
        s = d.get("confidence", d.get("score", 0.5))
        try: s = float(s)
        except: s = 0.5
        s = max(0.0, min(1.0, s))
        item = {"bbox":[x,y,w,h], "score":s}
        if name2id is not None:
            item["category_id"] = int(name2id[lab])
        else:
            item["label"] = lab
        raw.append(item)

    kept = [r for r in raw if r["score"] >= min_conf]
    # 类内 NMS（若有类别）
    if name2id is not None:
        by = {}
        for r in kept: by.setdefault(r["category_id"], []).append(r)
        kept2 = []
        for _, items in by.items(): kept2.extend(nms_xywh(items, nms_iou))
    else:
        kept2 = kept
    kept3 = nms_xywh(kept2, nms_iou)
    kept3 = sorted(kept3, key=lambda d: d["score"], reverse=True)[: max_objects]
    return kept3, self_acc, raw

# ========= 单图可视化 =========

def run_single(args, model, processor, classes: Optional[List[str]]=None):
    image = Image.open(args.image).convert("RGB"); W,H = image.size
    valid_names = set(c.lower() for c in classes) if classes else None
    name2id = None

    prompt = build_prompt(classes or ["person","car","bicycle","chair","cup","bottle","tv"], args.max_objects, args.self_report)
    print("[Gen] generating …")
    text = generate_text(model, processor, image, prompt, device=("cuda" if torch.cuda.is_available() else "cpu"),
                         max_new_tokens=args.tokens)
    print("[Gen] raw (head):", (text[:400] + ("..." if len(text)>400 else "")))

    kept, self_acc, _ = postprocess_one(text, W, H, valid_names, name2id,
                                        args.min_conf, args.nms_iou, args.max_objects, drop_giant=True)
    print(f"[Info] kept={len(kept)}; self_report={self_acc}")

    out_img = args.out_img or (os.path.splitext(args.image)[0] + "_llava_det.jpg")
    vis = draw_boxes(image, kept, lw=3); vis.save(out_img, quality=95)
    print("[Save] vis ->", out_img)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f: json.dump(kept, f, ensure_ascii=False, indent=2)
        print("[Save] json ->", args.out_json)
    if args.save_raw:
        with open(os.path.splitext(out_img)[0]+".raw.txt","w",encoding="utf-8") as f: f.write(text)

# ========= COCO eval =========

def run_coco(args, model, processor):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    print(f"[COCO] Anns: {args.val_ann}\n[COCO] Images dir: {args.val_img}")
    coco = COCO(args.val_ann)
    cats = coco.loadCats(coco.getCatIds())
    id2name = {c["id"]: c["name"].strip().lower() for c in cats}
    name2id = {v:k for k,v in id2name.items()}
    classes = [id2name[i] for i in sorted(id2name.keys())]
    valid_names = set(classes)

    img_ids = coco.getImgIds()
    if args.subset and args.subset>0: img_ids = img_ids[: args.subset]
    info = {im["id"]: im for im in coco.dataset["images"]}

    prompt = build_prompt(classes, args.max_objects, want_self_acc=False)
    dt: List[Dict[str,Any]] = []
    raw_dump: List[Tuple[int,str]] = []

    print(f"[Eval] Images: {len(img_ids)} | tokens={args.tokens} min_conf={args.min_conf} nms_iou={args.nms_iou} max_objects={args.max_objects}")
    t0=time.time()
    for i,img_id in enumerate(img_ids,1):
        meta = info[img_id]; W,H = int(meta["width"]), int(meta["height"])
        img_path = os.path.join(args.val_img, meta["file_name"])
        try: image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Warn] open failed {img_path} ({e})"); continue

        text = generate_text(model, processor, image, prompt, device=("cuda" if torch.cuda.is_available() else "cpu"),
                             max_new_tokens=args.tokens)
        kept, _, raw = postprocess_one(text, W, H, valid_names, name2id,
                                       args.min_conf, args.nms_iou, args.max_objects, drop_giant=True)
        dt.extend({"image_id": int(img_id), **d} for d in kept)
        if args.save_raw: raw_dump.append((img_id, text))
        if (i<=5) or (i % args.progress_every == 0):
            print(f"[{i}/{len(img_ids)}] dt={len(dt)} (raw this img: {len(raw)} -> kept {len(kept)})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as f: json.dump(dt, f)
    print(f"[Save] Detections -> {args.out} ({len(dt)} items)")
    if args.save_raw:
        with open(os.path.splitext(args.out)[0]+".raw.txt","w",encoding="utf-8") as f:
            for img_id,t in raw_dump: f.write(f"# image_id={img_id}\n{t}\n\n")

    if len(dt)==0:
        print("[Warn] No detections, skip COCOeval"); return
    cocoDt = coco.loadRes(args.out)
    e = COCOeval(coco, cocoDt, iouType="bbox")
    e.evaluate(); e.accumulate(); e.summarize()
    print("==== SUMMARY ====")
    print(f"mAP@[.50:.95]: {e.stats[0]:.4f}")
    print(f"AP50:          {e.stats[1]:.4f}")
    print(f"AR@100:        {e.stats[8]:.4f}")

# ========= main =========

def main():
    ap = argparse.ArgumentParser()
    # 模型
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--proc-dir", default=None, help="若合并模型目录缺 tokenizer，请指向基座目录")
    ap.add_argument("--device-map", choices=["auto","cuda","cpu"], default="auto")
    # 通用
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--min-conf", type=float, default=0.30)
    ap.add_argument("--nms-iou", type=float, default=0.60)
    ap.add_argument("--max-objects", type=int, default=12)
    # 单图
    ap.add_argument("--image", default=None)
    ap.add_argument("--self-report", action="store_true")
    ap.add_argument("--out-img", default=None)
    ap.add_argument("--out-json", default=None)
    # COCO
    ap.add_argument("--val-ann", default=None)
    ap.add_argument("--val-img", default=None)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--out", default="/tmp/llava_dt.json")
    ap.add_argument("--save-raw", action="store_true")
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device=="cuda" else torch.float32

    print("[Load] %s" % args.model_dir)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=dtype,
        device_map=("cuda:0" if (device=="cuda" and args.device_map=="cuda") else
                    ("auto" if device=="cuda" else None))
    )
    proc_dir = args.proc_dir or args.model_dir
    processor = AutoProcessor.from_pretrained(proc_dir)

    # 清理生成配置里的采样项
    try:
        gc = model.generation_config
        gc.do_sample=False; gc.num_beams=1
        for k in ("temperature","top_p","top_k","typical_p"):
            if hasattr(gc,k): setattr(gc,k, None)
    except: pass

    if args.image:
        # 若提供 COCO 标注，则用其类名约束；否则用常见类兜底
        classes = None
        if args.val_ann:
            from pycocotools.coco import COCO
            coco = COCO(args.val_ann)
            cats = coco.loadCats(coco.getCatIds())
            classes = [c["name"].strip().lower() for c in cats]
        run_single(args, model, processor, classes)
    elif args.val_ann and args.val_img:
        run_coco(args, model, processor)
    else:
        raise SystemExit("请指定 --image（单图）或 --val-ann + --val-img（COCO 评测）")

if __name__ == "__main__":
    main()
