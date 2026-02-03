# -*- coding: utf-8 -*-
import json
import re

_PREFIX_RE = re.compile(r"^(关键词|关键字)[:：]\s*", re.IGNORECASE)
_HINT_RE = re.compile(
    r"(关键词|关键字)\s*(?:应该是|应为|应包括|包括|是|为)?\s*[:：]?\s*",
    re.IGNORECASE,
)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
_MARKER_PLURAL_RE = re.compile(
    r"BEGIN\s*[_\s]*KEYWORDS\s*[:：]?\s*(.*?)\s*END\s*[_\s]*KEYWORDS",
    re.IGNORECASE | re.DOTALL,
)
_MARKER_SINGULAR_RE = re.compile(
    r"BEGIN\s*[_\s]*KEYWORD\s*[:：]?\s*(.*?)\s*END\s*[_\s]*KEYWORDS",
    re.IGNORECASE | re.DOTALL,
)
_BEGIN_ANY_RE = re.compile(r"BEGIN\s*[_\s]+\s*[A-Z0-9_]+", re.IGNORECASE | re.DOTALL)
_SEP_RE = re.compile(r"[;,\\n]+")


def normalize_separators(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return (
        text.replace("\u3000", " ")
        .replace("；", ";")
        .replace("，", ",")
        .replace("、", ",")
    )


def dedup_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if item is None:
            continue
        if not isinstance(item, str):
            item = str(item)
        item = item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def normalize_keyword_list(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        items = []
        for item in raw:
            if item is None:
                continue
            if not isinstance(item, str):
                item = str(item)
            item = normalize_separators(item).strip()
            if item:
                items.append(item)
        return dedup_keep_order(items)
    if not isinstance(raw, str):
        raw = str(raw)
    text = normalize_separators(raw).strip()
    text = _PREFIX_RE.sub("", text)
    parts = _SEP_RE.split(text)
    return dedup_keep_order(parts)


def extract_json_object(text):
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def last_match(regex, text):
    match = None
    for item in regex.finditer(text):
        match = item
    return match


def first_non_empty_line(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def parse_keywords_from_output(text):
    if text is None:
        return []
    if isinstance(text, list):
        return normalize_keyword_list(text)
    if not isinstance(text, str):
        text = str(text)
    text = _THINK_TAG_RE.sub("", text).strip()
    marker_match = last_match(_MARKER_PLURAL_RE, text)
    if marker_match:
        return normalize_keyword_list(first_non_empty_line(marker_match.group(1)))
    marker_match = last_match(_MARKER_SINGULAR_RE, text)
    if marker_match:
        return normalize_keyword_list(first_non_empty_line(marker_match.group(1)))
    begin_match = last_match(_BEGIN_ANY_RE, text)
    if begin_match:
        candidate = text[begin_match.end() :]
        line = first_non_empty_line(candidate)
        if line:
            return normalize_keyword_list(line)
    json_blob = extract_json_object(text)
    if json_blob:
        try:
            obj = json.loads(json_blob)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            for key in ("keywords", "keyword", "labels", "key_words", "keyWords", "关键词"):
                if key in obj:
                    return normalize_keyword_list(obj[key])
        elif isinstance(obj, list):
            return normalize_keyword_list(obj)
    hint_match = None
    for match in _HINT_RE.finditer(text):
        hint_match = match
    if hint_match:
        candidate = text[hint_match.end() :]
        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        if lines:
            candidate = lines[-1]
        candidate = candidate.strip(" ：:;；,，、。")
        return normalize_keyword_list(candidate)
    return normalize_keyword_list(text)


def compute_metrics(gold_list, pred_list, match_mode="strict"):
    return compute_metrics_with_mode(gold_list, pred_list, match_mode=match_mode)


def is_keyword_match(pred, gold, match_mode):
    if match_mode == "strict":
        return pred == gold
    if match_mode == "loose":
        return pred in gold or gold in pred
    raise ValueError(f"Unknown match_mode: {match_mode}")


def match_keywords(gold_list, pred_list, match_mode="strict"):
    gold_matched = [False] * len(gold_list)
    pred_matched = [False] * len(pred_list)
    for pred_idx, pred in enumerate(pred_list):
        for gold_idx, gold in enumerate(gold_list):
            if gold_matched[gold_idx]:
                continue
            if is_keyword_match(pred, gold, match_mode):
                pred_matched[pred_idx] = True
                gold_matched[gold_idx] = True
                break
    fp_items = [pred_list[i] for i, matched in enumerate(pred_matched) if not matched]
    fn_items = [gold_list[i] for i, matched in enumerate(gold_matched) if not matched]
    tp = sum(1 for matched in pred_matched if matched)
    fp = len(fp_items)
    fn = len(fn_items)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fp_items": fp_items,
        "fn_items": fn_items,
        "pred_matched": pred_matched,
        "gold_matched": gold_matched,
    }


def compute_metrics_with_mode(gold_list, pred_list, match_mode="strict"):
    stats = match_keywords(gold_list, pred_list, match_mode=match_mode)
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]
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
