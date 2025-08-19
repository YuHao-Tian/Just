#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-reported "accuracy" collector for LLaVA-1.5 (chat-template).
- 对每张图：要求模型输出 detections + {"self_report":{"accuracy": x}}
- 解析 x（支持 0~1 小数或百分比形式），汇总平均
- 仅用于演示/自检；不等价于 COCO 指标
"""

import os, re, json, argparse, time
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ---------- robust helpers ----------

def strip_code_fences(t: str) -> str:
    t = t.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t).strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    return t

def maybe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def parse_json_or_fragment(text: str) -> Optional[dict]:
    t = strip_code_fences(text)
    obj = maybe_json_load(t)
    base = t
    if obj is None:
        m = re.search(r"(\{[\s\S]*\})", t)  # 抓最外层 {}
        if m:
            base = m.group(1)
            obj = maybe_json_load(base)
        if obj is None:
            for suf in ("}", "}}", "]}", "}]}",):
                obj = maybe_json_load(base + suf)
                if obj is not None:
                    break
    return obj if isinstance(obj, dict) else None

def extract_accuracy(text: str) -> Optional[float]:
    """
    优先从 JSON 字段 self_report.accuracy 取；
    否则回退到正则提取数字/百分比；统一到 [0,1]
    """
    obj = parse_json_or_fragment(text)
    if obj:
        # 常见结构：{"self_report":{"accuracy":0.873}}
        cur = obj.get("self_report", {})
        if isinstance(cur, dict) and "accuracy" in cur:
            try:
                v = float(cur["accuracy"])
                if 0.0 <= v <= 1.0: return v
                if 1.0 < v <= 100.0: return v/100.0
            except Exception:
                pass
        # 退一步：在整个 JSON 字符串里搜 "accuracy": number
        m = re.search(r'"accuracy"\s*:\s*([0-9]+(?:\.[0-9]+)?)', json.dumps(obj))
        if m:
            v = float(m.group(1))
            return v/100.0 if v > 1.0 else max(0.0, min(1.0, v))

    # 最后：从原文本里搜
    t = strip_code_fences(text)
    # 百分比
    m = re.search(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*%', t)
    if m:
        v = float(m.group(1))
        if 0.0 <= v <= 100.0: return v/100.0
    # 小数 0.x 或 1.0
    m = re.search(r'(?<![0-9])((?:0?\.[0-9]+)|1(?:\.0+)?)', t)
    if m:
        v = float(m.group(1))
        return max(0.0, min(1.0, v))
    return None

# ---------- chat generation ----------

PROMPT_TEMPLATE = (
    "Perform object detection on this image. Then output ONLY ONE JSON object with two keys:\n"
    " 1) \"detections\": a list like [{\"label\":\"<category>\", \"box\":[x1,y1,x2,y2], \"confidence\":0.xx}],\n"
    "    coordinates normalized to [0,1] (top-left x1,y1; bottom-right x2,y2), up to {max_objects} items;\n"
    " 2) \"self_report\": {{\"accuracy\": A}}, where A is YOUR estimated overall detection accuracy for this image\n"
    "    as a float in [0,1], rounded to 3 decimals (or percentage like 87%).\n"
    "Use concise English category names (COCO-style if applicable). No markdown/code fences, no explanations."
)

def build_messages(prompt_text: str):
    # chat-template 输入：content 里先 image 再 text（不要在 text 里写 <image>）
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]

def generate_text(model, processor, image: Image.Image, prompt: str, device: str, max_new_tokens: int) -> str:
    messages = build_messages(prompt)
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
            pad_token_id=getattr(processor.tokenizer, "pad_token_id", None)
                        or getattr(processor.tokenizer, "eos_token_id", None),
        )
    seq = out.sequences[0]
    gen_ids = seq[inputs["input_ids"].shape[-1]:]  # 只解码新生成 tokens
    return processor.decode(gen_ids, skip_special_tokens=True).strip()

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--proc-dir", default=None, help="可选：处理器从这个目录加载（比如基座）")
    ap.add_argument("--val-img",   required=True, help="图像文件夹（如 COCO val2017）")
    ap.add_argument("--subset",    type=int, default=50, help="最多取多少张（按文件名排序前 N 张）")
    ap.add_argument("--tokens",    type=int, default=256)
    ap.add_argument("--max-objects", type=int, default=20)
    ap.add_argument("--out",       default="self_report_acc.json")
    ap.add_argument("--save-raw",  action="store_true")
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    print(f"[Load] model: {args.model_dir}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=dtype, device_map="auto" if device=="cuda" else None
    )
    proc_dir = args.proc_dir or args.model_dir
    processor = AutoProcessor.from_pretrained(proc_dir)

    # 清理 sampler 配置
    try:
        gc = model.generation_config
        gc.do_sample = False; gc.num_beams = 1
        for k in ("temperature","top_p","top_k","typical_p"):
            if hasattr(gc, k): setattr(gc, k, None)
    except Exception:
        pass

    # 收集文件
    files = [f for f in sorted(os.listdir(args.val_img)) if f.lower().endswith((".jpg",".jpeg",".png",".webp"))]
    if args.subset and args.subset > 0:
        files = files[: args.subset]
    print(f"[Info] images: {len(files)}")

    prompt = PROMPT_TEMPLATE.format(max_objects=args.max_objects)

    rows: List[Dict[str, Any]] = []
    accs: List[float] = []
    t0 = time.time()

    for i, fn in enumerate(files, 1):
        path = os.path.join(args.val_img, fn)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[Warn] open fail: {path} ({e})")
            continue

        text = generate_text(model, processor, img, prompt, device, args.tokens)
        acc = extract_accuracy(text)

        rows.append({
            "file_name": fn,
            "self_report_accuracy": acc,
            "raw": text if args.save_raw else None
        })
        if acc is not None:
            accs.append(acc)

        if (i <= 5) or (i % args.progress_every == 0):
            print(f"[{i}/{len(files)}] acc={acc}  (avg so far={sum(accs)/len(accs):.3f} if any)")

    # 保存
    out = {
        "model_dir": args.model_dir,
        "images_dir": args.val_img,
        "count": len(files),
        "avg_self_report_accuracy": (sum(accs)/len(accs)) if accs else None,
        "items": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[Save] -> {args.out}")
    print(f"[Done] avg_self_report_accuracy = {out['avg_self_report_accuracy']}, time={time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
