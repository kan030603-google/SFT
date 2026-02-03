# -*- coding: utf-8 -*-
import argparse
import json
import os

from lora_unsloth.keyword_eval_utils import (
    match_keywords,
    normalize_keyword_list,
    parse_keywords_from_output,
)


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_lists(obj):
    gold_raw = obj.get("gold_list", obj.get("gold"))
    pred_raw = obj.get("pred_list")
    if pred_raw is None:
        pred_raw = obj.get("pred")
    gold_list = normalize_keyword_list(gold_raw)
    if isinstance(pred_raw, list):
        pred_list = normalize_keyword_list(pred_raw)
    else:
        pred_list = parse_keywords_from_output(pred_raw)
    return gold_list, pred_list


def compute_metrics_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_file(path, label, topk, out_dir, match_mode):
    per_sample = []
    macro_sum = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    total = {"tp": 0, "fp": 0, "fn": 0}
    count = 0

    for obj in iter_jsonl(path):
        gold_list, pred_list = get_lists(obj)
        match = match_keywords(gold_list, pred_list, match_mode=match_mode)
        metrics = compute_metrics_from_counts(match["tp"], match["fp"], match["fn"])
        total["tp"] += metrics["tp"]
        total["fp"] += metrics["fp"]
        total["fn"] += metrics["fn"]
        macro_sum["precision"] += metrics["precision"]
        macro_sum["recall"] += metrics["recall"]
        macro_sum["f1"] += metrics["f1"]
        count += 1

        fp_items = match["fp_items"]
        fn_items = match["fn_items"]
        per_sample.append(
            {
                "id": obj.get("id"),
                "conversation_id": obj.get("conversation_id"),
                "fp_count": metrics["fp"],
                "fn_count": metrics["fn"],
                "fp": fp_items,
                "fn": fn_items,
                "pred_list": pred_list,
                "gold_list": gold_list,
                "input": obj.get("input"),
                "pred": obj.get("pred"),
                "gold": obj.get("gold"),
            }
        )

    macro = {
        "precision": macro_sum["precision"] / count if count else 0.0,
        "recall": macro_sum["recall"] / count if count else 0.0,
        "f1": macro_sum["f1"] / count if count else 0.0,
    }
    micro_precision = total["tp"] / (total["tp"] + total["fp"]) if (total["tp"] + total["fp"]) else 0.0
    micro_recall = total["tp"] / (total["tp"] + total["fn"]) if (total["tp"] + total["fn"]) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )
    micro = {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1}

    os.makedirs(out_dir, exist_ok=True)
    metrics_out = {
        "label": label,
        "file": os.path.abspath(path),
        "match_mode": match_mode,
        "samples": count,
        "tp": total["tp"],
        "fp": total["fp"],
        "fn": total["fn"],
        "macro": macro,
        "micro": micro,
    }
    metrics_path = os.path.join(out_dir, f"metrics_{label}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    top_fp = sorted(
        per_sample,
        key=lambda r: (-r["fp_count"], -r["fn_count"], str(r.get("id"))),
    )[:topk]
    top_fn = sorted(
        per_sample,
        key=lambda r: (-r["fn_count"], -r["fp_count"], str(r.get("id"))),
    )[:topk]
    top_fp_path = os.path.join(out_dir, f"top_fp_{label}.jsonl")
    top_fn_path = os.path.join(out_dir, f"top_fn_{label}.jsonl")
    with open(top_fp_path, "w", encoding="utf-8") as f:
        for row in top_fp:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(top_fn_path, "w", encoding="utf-8") as f:
        for row in top_fn:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metrics_out


def write_compare_tables(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for res in results:
        rows.append(
            {
                "label": res["label"],
                "samples": res["samples"],
                "tp": res["tp"],
                "fp": res["fp"],
                "fn": res["fn"],
                "macro_precision": res["macro"]["precision"],
                "macro_recall": res["macro"]["recall"],
                "macro_f1": res["macro"]["f1"],
                "micro_precision": res["micro"]["precision"],
                "micro_recall": res["micro"]["recall"],
                "micro_f1": res["micro"]["f1"],
            }
        )

    csv_path = os.path.join(out_dir, "compare_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        headers = list(rows[0].keys()) if rows else []
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")

    md_path = os.path.join(out_dir, "compare_metrics.md")
    with open(md_path, "w", encoding="utf-8") as f:
        if not rows:
            f.write("|label|\n|---|\n")
            return
        headers = list(rows[0].keys())
        f.write("|" + "|".join(headers) + "|\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            def fmt(v):
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)
            f.write("|" + "|".join(fmt(row[h]) for h in headers) + "|\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate keyword extraction predictions and compare models."
    )
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="Prediction JSONL path (repeatable).",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Label for each prediction file (repeatable).",
    )
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--out-dir", default="eval_results")
    parser.add_argument(
        "--match-mode",
        default="loose",
        choices=["loose", "strict"],
        help="Keyword matching mode.",
    )
    args = parser.parse_args()

    labels = args.label or []
    if labels and len(labels) != len(args.pred):
        raise SystemExit("Number of --label must match number of --pred.")
    if not labels:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in args.pred]

    results = []
    for path, label in zip(args.pred, labels):
        res = evaluate_file(path, label, args.topk, args.out_dir, args.match_mode)
        results.append(res)

    write_compare_tables(results, args.out_dir)

    for res in results:
        print(
            f"{res['label']}: macro_f1={res['macro']['f1']:.4f}, "
            f"micro_f1={res['micro']['f1']:.4f}, "
            f"tp={res['tp']}, fp={res['fp']}, fn={res['fn']}"
        )
    print(f"Saved compare table to {os.path.join(args.out_dir, 'compare_metrics.md')}")


if __name__ == "__main__":
    main()
