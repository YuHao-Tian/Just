import os, json, argparse, time, sys
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def try_parse_json(text: str):
    """Extract outermost JSON object if possible; else return empty schema."""
    try:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e+1])
    except Exception:
        pass
    return {"detections": []}

def build_canon_label(name2id):
    """Return a canonicalizer mapping raw labels to official COCO-80 names (lower-case)."""
    name_set = set(name2id.keys())
    no_space_map = {n.replace(" ", ""): n for n in name_set}
    syn = {
        "people":"person","man":"person","woman":"person","men":"person","women":"person",
        "boy":"person","girl":"person","kid":"person","child":"person","baby":"person",
        "motorbike":"motorcycle","aeroplane":"airplane","aircraft":"airplane",
        "trafficlight":"traffic light","traffic-light":"traffic light",
        "tvmonitor":"tv","tv monitor":"tv","television":"tv",
        "cellphone":"cell phone","mobile phone":"cell phone","smartphone":"cell phone","iphone":"cell phone",
        "sofa":"couch","pottedplant":"potted plant","potted plant":"potted plant",
        "diningtable":"dining table","hand bag":"handbag","hand-bag":"handbag",
        "wineglass":"wine glass","wine-glass":"wine glass",
        "tennis-racket":"tennis racket","baseball-bat":"baseball bat","baseball-glove":"baseball glove"
    }
    def canon(raw: str):
        s = (raw or "").lower().strip()
        s = s.replace("_"," ").replace("-"," ")
        while "  " in s: s = s.replace("  ", " ")
        if s in name_set: return s
        if s in syn: return syn[s]
        key = s.replace(" ","")
        if key in no_space_map: return no_space_map[key]
        return None
    return canon

def gen_and_decode_reply(model, proc, img, prompt, device, max_new_tokens=192):
    """Generate once and decode ONLY the assistant's newly generated text."""
    inputs = proc(images=img, text=prompt, return_tensors="pt")
    inputs = {k:(v.to(device) if hasattr(v,"to") else v) for k,v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True
        )
    seq = out.sequences[0]
    # 只解码新生成的部分
    gen_ids = seq[inputs["input_ids"].shape[-1]:]
    return proc.decode(gen_ids, skip_special_tokens=True).strip()

def labels_invalid(dets, allowed_lower: set):
    """Return True if any label is clearly invalid (placeholder or not in allowed set)."""
    for d in dets:
        lab = str(d.get("label","")).lower()
        if ("<" in lab) or (lab not in allowed_lower):
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/vipuser/llava-1.5-7b-hf")
    ap.add_argument("--val-ann",  default="/home/vipuser/coco/annotations/instances_val2017.json")
    ap.add_argument("--val-img",  default="/home/vipuser/coco/images/val2017")
    ap.add_argument("--subset",   type=int, default=10, help="evaluate first N images (0 = all 5000)")
    ap.add_argument("--tokens",   type=int, default=192)
    ap.add_argument("--out",      default="/home/vipuser/coco/llava_dt_base.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    # 1) Load base LLaVA-1.5-7B (no LoRA here)
    model = LlavaForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=dtype).to(device).eval()
    proc  = AutoProcessor.from_pretrained(args.model_dir)

    # 2) COCO classes & images
    coco = COCO(args.val_ann)
    cats = coco.loadCats(coco.getCatIds())
    class_names = [c["name"] for c in cats]              # official 80 names
    name2id = {c["name"].lower(): c["id"] for c in cats}
    canon_label = build_canon_label(name2id)
    allowed = set([n.lower() for n in class_names])

    imgs = coco.dataset["images"]
    ids  = [im["id"] for im in imgs]
    info = {im["id"]: (im["file_name"], im["width"], im["height"]) for im in imgs}
    if args.subset and args.subset > 0:
        ids = ids[:args.subset]
        print(f"[info] subset = {len(ids)} images")

    # 3) English instruction (NO placeholders, NO inline JSON example)
    instr = (
        "You are an object detection assistant. "
        "Return ONLY a valid JSON object with key 'detections'. "
        "Each item is: {\"label\": <one of the COCO-80 list>, \"box\": [x1,y1,x2,y2], \"confidence\": number in [0,1]}. "
        "Coordinates are normalized to [0,1] with (x1,y1)=top-left and (x2,y2)=bottom-right. "
        "If nothing is found, return {\"detections\":[]}. "
        "Use ONLY these labels (singular, English): " + ", ".join(class_names) + ". "
        "At most 8 objects.Round coordinates to 3 decimals. Output JSON only."
    )

    dt = []
    ok_json = 0
    t0 = time.time()

    for k, img_id in enumerate(ids, 1):
        fn, W, H = info[img_id]
        img_path = os.path.join(args.val_img, fn)
        img = Image.open(img_path).convert("RGB")

        # Build chat messages: one image + one text instruction
        messages = [{"role":"user","content":[
            {"type":"image"},
            {"type":"text","text": instr}
        ]}]
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True)

        # First try
        txt  = gen_and_decode_reply(model, proc, img, prompt, device, max_new_tokens=args.tokens)
        if k <= 5:
            print(f"==== RAW REPLY (image {k}: {fn}) ====")
            print(txt)

        data = try_parse_json(txt)

        # Retry if JSON invalid
        if not isinstance(data.get("detections", None), list):
            retry = [{"role":"user","content":[
                {"type":"image"},
                {"type":"text","text":"Only output valid JSON with the key 'detections'. No extra words."}
            ]}]
            rprompt = proc.apply_chat_template(retry, add_generation_prompt=True)
            txt2 = gen_and_decode_reply(model, proc, img, rprompt, device, max_new_tokens=128)
            if k <= 5:
                print(f"==== RETRY JSON (image {k}) ====")
                print(txt2)
            data = try_parse_json(txt2)

        dets = data.get("detections", [])
        # Retry if labels look invalid (e.g., placeholders or not in COCO-80)
        if isinstance(dets, list) and labels_invalid(dets, allowed):
            retry2 = [{"role":"user","content":[
                {"type":"image"},
                {"type":"text","text":
                 "Rewrite as JSON using ONLY these labels: " + ", ".join(class_names) +
                 ". Do NOT use placeholders. Output JSON only."}
            ]}]
            rprompt2 = proc.apply_chat_template(retry2, add_generation_prompt=True)
            txt_fix = gen_and_decode_reply(model, proc, img, rprompt2, device, max_new_tokens=128)
            if k <= 5:
                print(f"==== RETRY LABELS (image {k}) ====")
                print(txt_fix)
            data = try_parse_json(txt_fix)
            dets = data.get("detections", [])

        # Count JSON-parsable replies
        if isinstance(dets, list):
            ok_json += 1

        # Convert normalized boxes to COCO bbox and append detections
        for d in dets[:15]:
            if not isinstance(d, dict) or "box" not in d:
                continue
            lab = canon_label(d.get("label", ""))
            if lab is None:
                continue
            try:
                x1, y1, x2, y2 = d["box"]
            except Exception:
                continue
            x, y, w, h = x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H
            dt.append({
                "image_id": int(img_id),
                "category_id": name2id[lab],
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(d.get("confidence", 0.9))
            })

        print(f"[{k}/{len(ids)}] dt={len(dt)}")

    # Save detections & report JSON success rate
    with open(args.out, "w") as f:
        json.dump(dt, f)
    print(f"[saved] detections -> {args.out}")
    jsucc = ok_json / len(ids) if len(ids) else 0.0
    print(f"[info] JSON success rate: {ok_json}/{len(ids)} = {jsucc:.2%}")
    print(f"[time] processed {len(ids)} images in {time.time()-t0:.1f}s")

    # COCO evaluation
    if len(dt) == 0:
        print("[warn] No detections recorded. mAP will be 0.0. Check RAW/RETRY outputs above.")
        sys.exit(0)

    res = coco.loadRes(args.out)
    e = COCOeval(coco, res, "bbox")
    e.evaluate(); e.accumulate(); e.summarize()
    mAP   = float(e.stats[0])   # AP@[.50:.95]
    AP50  = float(e.stats[1])   # AP@0.50
    AR100 = float(e.stats[8])   # AR@100
    print(f"[metrics] mAP@[.50:.95]={mAP:.4f} | AP50={AP50:.4f} | AR@100={AR100:.4f}")

if __name__ == "__main__":
    main()
