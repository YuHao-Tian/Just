#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, argparse, time, math, random
from collections import defaultdict
from typing import List, Dict, Any

import torch
from PIL import Image

from transformers import AutoProcessor, LlavaForConditionalGeneration, StoppingCriteria, StoppingCriteriaList

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# -----------------------------
# Utilities
# -----------------------------

def set_torch_flags():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def try_float(v):
    try:
        if isinstance(v, list) and len(v) == 1:
            v = v[0]
        return float(v)
    except Exception:
        return None

def json_substring(s: str) -> str:
    """return the substring between last matching {...} if exists"""
    r = s.rfind("}")
    if r == -1:
        return s
    l = s.rfind("{", 0, r + 1)
    if l == -1:
        return s
    return s[l : r + 1]

def try_parse_json(s: str) -> Dict[str, Any]:
    s2 = json_substring(s.strip())
    try:
        obj = json.loads(s2)
        if isinstance(obj, dict) and "detections" in obj and isinstance(obj["detections"], list):
            return obj
    except Exception:
        pass
    return {"detections": []}

def group_topk(dets: List[Dict[str, Any]], k_total=6, k_per_class=2) -> List[Dict[str, Any]]:
    by_cls = defaultdict(list)
    for d in dets:
        lbl = str(d.get("label", "")).strip()
        sc  = d.get("confidence", 0.0)
        if not isinstance(sc, (int, float)):
            continue
        by_cls[lbl].append((float(sc), d))
    kept = []
    for lbl, arr in by_cls.items():
        arr.sort(key=lambda x: x[0], reverse=True)
        kept.extend([d for _, d in arr[:k_per_class]])
    kept.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    return kept[:k_total]

class StopOnBrace(StoppingCriteria):
    """Stop as soon as the model outputs a '}' (optionally followed by newline)."""
    def __init__(self, tokenizer):
        self.ids = tokenizer.encode("}", add_special_tokens=False)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        ids = input_ids[0].tolist()
        L = len(self.ids)
        return L > 0 and ids[-L:] == self.ids


# -----------------------------
# Main eval
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--val-ann",   type=str, required=True)
    parser.add_argument("--val-img",   type=str, required=True)
    parser.add_argument("--subset",    type=int, default=500)
    parser.add_argument("--tokens",    type=int, default=512)
    parser.add_argument("--out",       type=str, required=True)
    parser.add_argument("--stop_on_brace", action="store_true", help="stop when '}' is produced")
    args = parser.parse_args()

    set_torch_flags()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    print(f"[info] device={device}, dtype={dtype}")

    print("[info] loading model & processor ...")
    model = LlavaForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=dtype).to(device)
    proc  = AutoProcessor.from_pretrained(args.model_dir)

    # Compose instruction (short & strict)
    class_names = [
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
        "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
        "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
        "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
        "clock","vase","scissors","teddy bear","hair drier","toothbrush"
    ]
    instr = (
        "You are an object detection assistant. "
        "Return ONLY a valid JSON object with key 'detections'. "
        "Each item: {\"label\": <one of the COCO-80 list>, \"box\": [x1,y1,x2,y2], \"confidence\": <0..1>}. "
        "Coordinates are normalized to [0,1] with (x1,y1)=top-left and (x2,y2)=bottom-right. "
        "If nothing is found, return {\"detections\": []}. "
        "Use ONLY these labels (singular, English): " + ", ".join(class_names) + ". "
        "Return at most 6 objects TOTAL. If many boxes of the same class, keep only the top 2 by confidence. "
        "Round numbers to 2 decimals. Output JSON only. End immediately after the closing '}'."
    )

    # COCO data
    coco = COCO(args.val_ann)
    img_ids = coco.getImgIds()
    if args.subset > 0:
        random.seed(0)
        img_ids = img_ids[:args.subset]
    print(f"[info] subset = {len(img_ids)} images")

    # name -> cat_id map using COCO categories
    name_to_cid = {}
    for cat in coco.loadCats(coco.getCatIds()):
        name_to_cid[cat["name"]] = cat["id"]

    # prepare stopping criteria (optional)
    stops = StoppingCriteriaList([StopOnBrace(proc.tokenizer)]) if args.stop_on_brace else None

    results = []
    ok_json = 0
    t0 = time.time()

    model.eval()
    for i, im_id in enumerate(img_ids, 1):
        info = coco.loadImgs([im_id])[0]
        img_path = os.path.join(args.val_img, info["file_name"])
        img = load_image(img_path)
        W, H = img.size

        # messages with single image + instruction
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instr}
                ]
            }
        ]

        # build chat prompt
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=prompt, images=img, return_tensors="pt").to(device)

        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=args.tokens,
                do_sample=False, temperature=0.0, top_p=1.0,
                repetition_penalty=1.15,
                no_repeat_ngram_size=16,
                eos_token_id=proc.tokenizer.eos_token_id,
                return_dict_in_generate=True
            )
            if stops is not None:
                gen_kwargs["stopping_criteria"] = stops

            out = model.generate(**inputs, **gen_kwargs)

        # only decode new tokens
        seq = out.sequences[0]
        new_tokens = seq[inputs["input_ids"].shape[1]:]
        txt = proc.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # parse with repair
        src_txt = txt
        data = try_parse_json(txt)

        # post-filter: keep top-2 per class, top-6 overall
        dets = data.get("detections", [])
        dets = group_topk(dets, k_total=6, k_per_class=2)

        # count json success (only when braces existed AND list parsed)
        if ("{" in src_txt and "}" in src_txt) and isinstance(dets, list):
            ok_json += 1

        # convert to COCO dt entries
        dt_this = 0
        for d in dets:
            lbl = str(d.get("label", "")).strip()
            cid_list = coco.getCatIds(catNms=[lbl]) if lbl else []
            if not cid_list:
                continue
            cid = cid_list[0]
            box = d.get("box", None)
            conf = try_float(d.get("confidence", 0.0))
            if box is None or conf is None:
                continue
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                continue
            x1 = clamp01(try_float(box[0]))
            y1 = clamp01(try_float(box[1]))
            x2 = clamp01(try_float(box[2]))
            y2 = clamp01(try_float(box[3]))
            if None in (x1, y1, x2, y2):
                continue
            # to xywh in absolute pixels
            x1a = max(0.0, min(W-1.0, x1 * W))
            y1a = max(0.0, min(H-1.0, y1 * H))
            x2a = max(0.0, min(W-1.0, x2 * W))
            y2a = max(0.0, min(H-1.0, y2 * H))
            w = max(0.0, x2a - x1a)
            h = max(0.0, y2a - y1a)
            results.append({
                "image_id": im_id,
                "category_id": cid,
                "bbox": [x1a, y1a, w, h],
                "score": float(conf)
            })
            dt_this += 1

        print(f"[{i}/{len(img_ids)}] dt={dt_this}")

    # save dt json
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"[saved] detections -> {args.out}")
    print(f"[info] JSON success rate: {ok_json}/{len(img_ids)} = {ok_json/len(img_ids)*100:.2f}%")
    print(f"[time] processed {len(img_ids)} images in {time.time()-t0:.1f}s")

    # Evaluate with COCO
    if len(results) == 0:
        print("[warn] empty results; skip COCOeval.")
        return

    coco_dt = coco.loadRes(args.out)
    coco_eval = COCOeval(coco, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # also print a concise line
    stats = coco_eval.stats
    print(f"[metrics] mAP@[.50:.95]={stats[0]:.4f} | AP50={stats[1]:.4f} | AR@100={stats[8]:.4f}")


if __name__ == "__main__":
    main()
