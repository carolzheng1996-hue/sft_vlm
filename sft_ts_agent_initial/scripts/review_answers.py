#!/usr/bin/env python3
"""Review generated SFT answers with deterministic checks and optional LLM judges."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import urllib.request
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def post_chat_completion(base_url: str, api_key: str, model: str, messages: list[dict[str, str]], timeout: int) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]


def assistant_json(record: dict[str, Any]) -> dict[str, Any]:
    content = record["messages"][-1]["content"]
    try:
        return json.loads(content)
    except Exception:
        return {"answer": content}


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def deterministic_review(record: dict[str, Any]) -> dict[str, Any]:
    obj = assistant_json(record)
    text = flatten_text(obj)
    expected_tools = record.get("expert_trace", {}).get("expected_tools", [])
    must_cover = record.get("expert_trace", {}).get("answer_must_cover", [])
    input_mode = record.get("metadata", {}).get("input_mode") or record.get("input_mode")

    tool_plan = obj.get("tool_plan", [])
    mentioned_tools = set()
    if isinstance(tool_plan, list):
        for step in tool_plan:
            if isinstance(step, dict) and step.get("tool"):
                mentioned_tools.add(str(step["tool"]))
    for tool in expected_tools:
        if tool in text:
            mentioned_tools.add(tool)

    expected_tool_coverage = 1.0
    if expected_tools:
        expected_tool_coverage = len([t for t in expected_tools if t in mentioned_tools or t in text]) / len(expected_tools)

    must_cover_hits = 0
    for key in must_cover:
        normalized = str(key).replace("_", " ")
        if key in text or normalized in text:
            must_cover_hits += 1
    must_cover_score = must_cover_hits / len(must_cover) if must_cover else 1.0

    required_keys = ["tool_plan", "answer", "model_or_strategy_recommendation", "risks_and_uncertainties", "quality_checks"]
    schema_score = len([k for k in required_keys if k in obj]) / len(required_keys)

    answer_len = len(flatten_text(obj.get("answer", "")))
    length_score = 1.0 if 250 <= answer_len <= 2500 else 0.55 if answer_len >= 120 else 0.25

    image_score = 1.0
    if input_mode == "image_text":
        image_terms = ["图", "图像", "可视化", "overview", "seasonal", "heatmap", "residual", "从图"]
        image_score = 1.0 if any(term in text for term in image_terms) else 0.2

    complexity_signals = ["优先", "不应", "如果", "因为", "验证", "风险", "下一步", "回测", "指标"]
    complexity_score = min(1.0, sum(1 for s in complexity_signals if s in text) / 6)

    score = (
        0.22 * schema_score
        + 0.22 * expected_tool_coverage
        + 0.18 * must_cover_score
        + 0.14 * image_score
        + 0.12 * length_score
        + 0.12 * complexity_score
    )
    flags = []
    if schema_score < 1:
        flags.append("schema_incomplete")
    if expected_tool_coverage < 0.6:
        flags.append("missing_expected_tools")
    if must_cover_score < 0.45:
        flags.append("missing_required_aspects")
    if image_score < 1:
        flags.append("image_evidence_missing")
    if length_score < 0.6:
        flags.append("too_short_or_too_long")
    if complexity_score < 0.5:
        flags.append("possibly_too_simple")

    if score < 0.58:
        label = "reply_error"
    elif "possibly_too_simple" in flags or must_cover_score < 0.65:
        label = "question_too_simple_or_answer_shallow"
    else:
        label = "pass"

    return {
        "deterministic_score": round(score, 4),
        "label": label,
        "flags": flags,
        "subscores": {
            "schema": round(schema_score, 4),
            "expected_tool_coverage": round(expected_tool_coverage, 4),
            "must_cover": round(must_cover_score, 4),
            "image_evidence": round(image_score, 4),
            "length": round(length_score, 4),
            "complexity": round(complexity_score, 4),
        },
    }


def judge_messages(record: dict[str, Any], det: dict[str, Any]) -> list[dict[str, str]]:
    prompt = {
        "instruction": "请作为时序/VLM SFT 数据评审专家，对样本质量打分。只输出合法 JSON。",
        "score_dimensions": {
            "correctness": "0-5，答案是否符合时序分析事实和可见证据",
            "tool_routing": "0-5，工具选择和调用顺序是否合理",
            "image_grounding": "0-5，image_text 样本是否真的结合图像；text_only 样本是否没有过度声称",
            "model_selection": "0-5，模型选择是否结合数据模式、任务、验证、成本",
            "sft_value": "0-5，是否值得用于 SFT，难度是否足够且不模板化",
        },
        "labels": ["pass", "reply_error", "question_too_simple", "needs_rewrite"],
        "deterministic_review": det,
        "sample": {
            "id": record.get("id"),
            "metadata": record.get("metadata"),
            "expert_trace": record.get("expert_trace"),
            "images": record.get("images"),
            "messages": record.get("messages"),
        },
        "required_output_schema": {
            "scores": {
                "correctness": 0,
                "tool_routing": 0,
                "image_grounding": 0,
                "model_selection": 0,
                "sft_value": 0,
            },
            "overall_score": 0,
            "label": "pass|reply_error|question_too_simple|needs_rewrite",
            "major_issues": [],
            "fix_suggestion": "如何回炉再造",
        },
    }
    return [
        {"role": "system", "content": "你是严格的数据质量评审员。只输出合法 JSON。"},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, indent=2)},
    ]


def parse_judge(raw: str) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
    except Exception:
        return {"overall_score": 0, "label": "needs_rewrite", "major_issues": ["judge_invalid_json"], "raw": raw}
    return obj


def aggregate(det: dict[str, Any], judges: list[dict[str, Any]]) -> dict[str, Any]:
    if not judges:
        return {
            "final_score": det["deterministic_score"],
            "final_label": det["label"],
        }
    scores = [float(j.get("overall_score", 0)) / 5 for j in judges]
    avg = statistics.mean(scores)
    final_score = 0.45 * det["deterministic_score"] + 0.55 * avg
    labels = [j.get("label") for j in judges]
    if det["label"] == "reply_error" or labels.count("reply_error") >= max(1, len(labels) // 2 + 1):
        final_label = "reply_error"
    elif "needs_rewrite" in labels or det["label"] == "question_too_simple_or_answer_shallow":
        final_label = "needs_rewrite"
    elif labels.count("question_too_simple") >= 1:
        final_label = "question_too_simple"
    else:
        final_label = "pass" if final_score >= 0.72 else "needs_rewrite"
    return {"final_score": round(final_score, 4), "final_label": final_label}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-models", default="")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    judge_models = [m.strip() for m in args.judge_models.split(",") if m.strip()]
    api_key = os.environ.get(args.api_key_env, "")
    if judge_models and not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    records = list(iter_jsonl(args.input))
    if args.limit is not None:
        records = records[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for idx, record in enumerate(records, start=1):
            det = deterministic_review(record)
            judge_results = []
            for model in judge_models:
                try:
                    raw = post_chat_completion(args.base_url, api_key, model, judge_messages(record, det), args.timeout)
                    judge_results.append({"model": model, **parse_judge(raw)})
                except Exception as e:
                    judge_results.append({"model": model, "overall_score": 0, "label": "needs_rewrite", "major_issues": [str(e)]})
            agg = aggregate(det, judge_results)
            row = {
                "id": record.get("id"),
                "metadata": record.get("metadata"),
                "deterministic_review": det,
                "judge_reviews": judge_results,
                **agg,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{idx}/{len(records)}] {record.get('id')} {agg['final_label']} {agg['final_score']}")


if __name__ == "__main__":
    main()
