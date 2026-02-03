# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from unsloth import FastLanguageModel

from lora_unsloth.keyword_eval_utils import (
    normalize_keyword_list,
    parse_keywords_from_output,
)


DEFAULT_MODEL_PATH = "D:/DATA/MODEL/qwen/unsloth/Qwen3-8B-unsloth-bnb-4bit"
DEFAULT_ADAPTER_PATH = "D:/DATA/MODEL/LoRA2qwen/outputs/Qwen3-8B-sft-lora-adapter-unsloth"
DEFAULT_INPUT_PATH = "data/keywords_data_test.jsonl"
DEFAULT_SYSTEM_PROMPT = (
    "你是关键词抽取助手。\n"
    "只输出标记区间中的关键词列表，不要输出任何解释或多余文字。\n"
    "输出格式固定为：\n"
    "BEGIN_KEYWORDS\n"
    "kw1;kw2;kw3\n"
    "END_KEYWORDS\n"
    "关键词要求：\n"
    "1) 每个词必须是原文中的连续子串，不得改写或编造。\n"
    "2) 禁止输出描述性短语，如“结论”“方法”“提示”“部分提到”等。\n"
    "3) 输出数量限制为 3-8 个，按重要性排序。若不足 3 个，只输出真实存在的子串。\n"
    "\n"
    "示例1\n"
    "输入：关键词识别：\n"
    "题目：锂离子电池快充机理\n"
    "摘要：本文研究快充过程中锂析出与电解液分解，提出石墨负极表面改性和电解液添加剂策略。\n"
    "输出：\n"
    "BEGIN_KEYWORDS\n"
    "锂离子电池;快充;锂析出;石墨负极;电解液添加剂\n"
    "END_KEYWORDS\n"
    "\n"
    "示例2\n"
    "输入：关键词识别：\n"
    "目的 研究不同年龄心肌炎患儿的心率及心率变异性(HRV)改变。\n"
    "方法 对120例心肌炎患儿进行24 h全程动态心电图检查。\n"
    "结论 心肌炎患儿的HRV普遍降低。\n"
    "输出：\n"
    "BEGIN_KEYWORDS\n"
    "心肌炎患儿;心率;心率变异性;HRV;动态心电图检查\n"
    "END_KEYWORDS\n"
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def iter_jsonl(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield idx, json.loads(line)


def iter_conversations(obj):
    conversations = obj.get("conversation") or []
    if isinstance(conversations, dict):
        yield 0, conversations
        return
    if not isinstance(conversations, list):
        return
    for idx, item in enumerate(conversations):
        if isinstance(item, dict):
            yield idx, item
        elif isinstance(item, list):
            for jdx, sub in enumerate(item):
                if isinstance(sub, dict):
                    yield f"{idx}-{jdx}", sub


def load_model_and_tokenizer(args):
    device_map = args.device_map
    if device_map is None:
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        device_map=device_map,
        dtype=None,
        load_in_4bit=not args.no_load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    if args.enable_adapter:
        if not args.adapter_path:
            raise ValueError("--enable-adapter requires --adapter-path")
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter_path)
        except Exception:
            if hasattr(model, "load_adapter"):
                model.load_adapter(args.adapter_path)
            else:
                raise
    FastLanguageModel.for_inference(model)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer, device_map


def build_prompt(tokenizer, system_prompt, user_text):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )


def generate_one(model, tokenizer, prompt, args):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run keyword extraction inference with a base or SFT model."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument(
        "--enable-adapter",
        action="store_true",
        help="Enable LoRA adapter loading; requires --adapter-path.",
        default=False
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--no-load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT

    set_seed(args.seed)
    model, tokenizer, device_map = load_model_and_tokenizer(args)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    meta = {
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(args.output),
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "enable_adapter": args.enable_adapter,
        "seed": args.seed,
        "system_prompt": system_prompt,
        "gen_params": {
            "max_seq_length": args.max_seq_length,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "load_in_4bit": not args.no_load_in_4bit,
            "load_in_8bit": args.load_in_8bit,
            "device_map": device_map,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.output + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    total = 0
    start = time.time()
    with open(args.output, "w", encoding="utf-8") as f:
        for line_idx, obj in iter_jsonl(args.input, limit=args.limit):
            base_id = obj.get("conversation_id", line_idx)
            for conv_idx, conv in iter_conversations(obj):
                human = (conv.get("human") or "").strip()
                assistant = (conv.get("assistant") or "").strip()
                if not human:
                    continue
                prompt = build_prompt(tokenizer, system_prompt, human)
                pred_text = generate_one(model, tokenizer, prompt, args)
                pred_text = pred_text.strip()
                pred_list = parse_keywords_from_output(pred_text)
                record_id = f"{base_id}_{conv_idx}"
                gold_list = normalize_keyword_list(assistant)
                out = {
                    "id": record_id,
                    "conversation_id": base_id,
                    "category": obj.get("category"),
                    "dataset": obj.get("dataset"),
                    "input": human,
                    "gold": assistant,
                    "gold_list": gold_list,
                    "pred": pred_text.strip(),
                    "pred_list": pred_list,
                    "model_path": args.model_path,
                    "adapter_path": args.adapter_path,
                    "enable_adapter": args.enable_adapter,
                    "seed": args.seed,
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                total += 1
                if args.log_every and total % args.log_every == 0:
                    elapsed = time.time() - start
                    print(f"Processed {total} samples in {elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"Done. Wrote {total} samples to {args.output} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
