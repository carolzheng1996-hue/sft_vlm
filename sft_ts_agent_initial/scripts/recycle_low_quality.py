#!/usr/bin/env python3
"""Create second-pass seed questions for low-quality or too-easy SFT samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in iter_jsonl(path)}


def critique_text(review: dict[str, Any]) -> str:
    det = review.get("deterministic_review", {})
    flags = det.get("flags", [])
    judge_issues = []
    for judge in review.get("judge_reviews", []):
        judge_issues.extend(judge.get("major_issues", []))
        if judge.get("fix_suggestion"):
            judge_issues.append(judge["fix_suggestion"])
    parts = [
        f"上一版 final_label={review.get('final_label')} final_score={review.get('final_score')}",
        f"deterministic_flags={flags}",
    ]
    if judge_issues:
        parts.append("judge_issues=" + "；".join(str(x) for x in judge_issues[:8]))
    return "\n".join(parts)


def strengthened_instruction(label: str) -> str:
    if label == "question_too_simple":
        return (
            "请把问题升级为更有区分度的专家任务：必须要求比较至少两种工具路线或模型路线，"
            "加入一个容易误判的反例，并要求说明为什么不能只凭单一统计量或单张图下结论。"
        )
    if label == "reply_error":
        return (
            "请针对上一版错误重做答案：必须逐项覆盖 expected_tools 和 answer_must_cover，"
            "不得臆造具体数值；如果证据不足，用待确认项表达。"
        )
    return (
        "请回炉改写为更稳定的 SFT 样本：问题更具体，答案更可验证，工具调用和模型选择理由更清晰。"
    )


def recycle_seed(seed: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    label = review.get("final_label", "needs_rewrite")
    new_seed = json.loads(json.dumps(seed, ensure_ascii=False))
    new_seed["id"] = seed["id"] + "_recycle_v2"
    new_seed["metadata"] = {
        **seed.get("metadata", {}),
        "recycled_from": seed["id"],
        "recycle_label": label,
        "recycle_score": review.get("final_score"),
    }
    new_seed["expert_trace"] = {
        **seed.get("expert_trace", {}),
        "recycle_critique": critique_text(review),
    }
    new_seed["question"] = seed.get("question", "") + "\n\n回炉要求：" + strengthened_instruction(label)

    if new_seed.get("messages") and len(new_seed["messages"]) >= 2:
        original = new_seed["messages"][1]["content"]
        recycle_payload = {
            "recycle_instruction": strengthened_instruction(label),
            "critique": critique_text(review),
            "required_changes": [
                "保留原始 case 和可见文件约束",
                "提高问题难度或修复答案错误",
                "答案必须包含 tool_plan、answer、model_or_strategy_recommendation、risks_and_uncertainties、quality_checks",
                "image_text 样本必须说明图像证据的作用",
            ],
        }
        new_seed["messages"][1]["content"] = (
            original
            + "\n\n【回炉再造约束】\n"
            + json.dumps(recycle_payload, ensure_ascii=False, indent=2)
        )
    return new_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", default="reply_error,needs_rewrite,question_too_simple")
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    seeds = load_jsonl_map(args.seeds)
    labels = {x.strip() for x in args.labels.split(",") if x.strip()}
    selected = []
    for review in iter_jsonl(args.reviews):
        if review.get("final_label") not in labels:
            continue
        seed_id = review.get("id")
        seed = seeds.get(seed_id)
        if not seed:
            continue
        selected.append(recycle_seed(seed, review))
        if args.max_items is not None and len(selected) >= args.max_items:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"output": str(args.out), "num_recycled": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
