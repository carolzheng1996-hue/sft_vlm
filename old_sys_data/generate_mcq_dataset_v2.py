#!/usr/bin/env python3
"""Build a less leaky MCQ-with-series benchmark variant.

This script intentionally keeps answer-side fields in the generated JSONL for
human inspection. The model-facing prompt lives under ``input`` and is rewritten
to avoid leaking chart-derived evidence through text-only observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


CHOICE_IDS = ["A", "B", "C", "D"]


TASK_VISUAL_EVIDENCE = {
    "复杂时序特性的模型推荐": "从图中判断是否同时存在多尺度周期、缓慢漂移和局部脉冲，再选择建模方案。",
    "预测结果 Bad Case 根因分析": "从真实值、预测值和误差形态中判断失败类型，例如滞后、峰值低估或相位错位。",
    "多变量关联与异常定位": "从多变量曲线的先后关系、持续时间和恢复情况判断是故障还是可恢复负载波动。",
    "突变点检测与响应": "从曲线中区分短暂异常、持续 level shift 和季节强度变化。",
    "长依赖周期识别": "从周期重复形态判断短周期与长周期是否都需要建模。",
    "区间预测评估": "从预测区间与真实值的相对位置判断覆盖不足、过宽或条件化失配。",
    "残差诊断与遗漏因素发现": "从残差时间图、ACF 或分组残差中判断是否仍有未解释结构。",
    "间歇性需求预测与指标选择": "从零值块、偶发需求峰和批量需求形态判断建模与指标选择。",
    "短历史新品预测方案设计": "从短历史走势和可迁移同类模式判断冷启动方案。",
    "促销外生变量与未来信息泄漏": "从促销前后走势和候选特征语义判断哪些变量预测时可得。",
    "验证方案与数据泄漏判断": "从切分方式、预处理时点和特征构造方式判断是否模拟真实预测 cutoff。",
    "多模型上线选择与业务权衡": "从指标、延迟、稳定性和高峰误差表现综合判断上线模型。",
    "层级预测与加总一致性": "从层级结构和各层误差表现判断是否需要 reconciliation。",
    "线上监控、漂移检测与重训练策略": "从误差、输入质量和业务切片走势判断漂移根因与处置顺序。",
}


TASK_SHORT_CONTEXT = {
    "复杂时序特性的模型推荐": "系统采集到一段业务核心指标历史序列，需要为未来预测选择合适模型。",
    "预测结果 Bad Case 根因分析": "已有模型在某个验证窗口出现明显误差，需要定位 bad case 根因。",
    "多变量关联与异常定位": "监控系统给出多变量时间序列告警，需要判断是故障还是正常扰动。",
    "突变点检测与响应": "线上指标疑似发生分布变化，需要判断变化形态并给出响应策略。",
    "长依赖周期识别": "业务希望识别序列中的主要长短周期，并选择能覆盖上下文的模型。",
    "区间预测评估": "业务使用预测区间进行容量或库存决策，需要判断区间是否校准。",
    "残差诊断与遗漏因素发现": "模型已完成训练，需要通过预测结果和残差诊断判断是否遗漏结构。",
    "间歇性需求预测与指标选择": "备件或低频 SKU 需求极不连续，需要选择合适模型和指标。",
    "短历史新品预测方案设计": "新品历史较短，但存在同类商品历史，需要设计冷启动预测方案。",
    "促销外生变量与未来信息泄漏": "团队准备用促销相关变量预测未来销量，需要审计特征可得性。",
    "验证方案与数据泄漏判断": "团队汇报离线指标很好，需要判断验证方案是否存在时间泄漏。",
    "多模型上线选择与业务权衡": "多个候选模型的指标、延迟和可解释性不同，需要选择上线方案。",
    "层级预测与加总一致性": "企业需要同时预测多个层级，并保证上下级预测口径一致。",
    "线上监控、漂移检测与重训练策略": "模型上线后误差升高，需要设计监控、根因定位和重训练策略。",
}


PADDING_CLAUSES = [
    "同时需要用滚动回测确认该判断，而不能只看单点误差。",
    "但仍要检查业务切片、预测时点可得性和上线维护成本。",
    "并且应把该方案与简单基线放在同一 forecast horizon 下比较。",
    "同时需要确认它不会引入未来信息或掩盖关键风险场景。",
    "但要结合图中局部形态、异常窗口和稳定性证据再决策。",
    "并在高峰、低谷、异常和分布变化窗口分别做误差切片。",
]

LONG_NEUTRAL_SUFFIXES = [
    "作答时还需要核验图中局部形态、预测时点可得信息、业务切片误差以及是否会掩盖关键风险场景。",
    "作答时应同时检查该判断是否符合图中证据、真实 forecast horizon、上线约束和数据泄漏审计要求。",
    "作答时还要对照简单基线、异常窗口、高峰低谷表现和预测时可获得的外生信息进行复核。",
    "作答时需要结合可视化证据、滚动回测设计、业务成本不对称和线上监控可维护性综合判断。",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_numeric_observation(question: str) -> str:
    """Keep only coarse numeric scale information from the old observation."""
    match = re.search(r"长度=([^，]+)，均值=([^，]+)，标准差=([^，；]+)", question)
    if not match:
        return "文本仅给出业务背景、任务目标和候选方案；关键时序形态需要从图像或原始序列中判断。"
    length, avg, std = match.groups()
    return (
        f"长度约为 {length}，均值约为 {avg}，标准差约为 {std}。"
        "文本不提供周期、异常位置、突变点或变量先后关系等已解析结论。"
    )


def make_prompt(record: dict[str, Any], mode: str) -> dict[str, Any]:
    design = record.get("benchmark_design", {})
    task = record["task_type"]
    context = TASK_SHORT_CONTEXT.get(task, design.get("business_context", "时序建模评测任务"))
    domain = record.get("domain") or design.get("business_context", "业务时序")
    target = design.get("target", "完成时序建模分析并给出建议")
    constraints = design.get("constraints", "需要结合业务目标、验证风险和落地成本")
    numeric = compact_numeric_observation(record["input"]["text_only"]["question"])
    visual = TASK_VISUAL_EVIDENCE.get(task, "请从图中读取关键时序证据。")

    lines = [
        f"背景：{domain}。",
        f"任务：{context}",
        f"目标：{target}",
        f"约束：{constraints}",
        "",
        "可用文本观测：",
        numeric,
        "",
    ]
    if mode == "multimodal":
        lines.extend(
            [
                f"请同时参考图片：{record['input']['multimodal']['image']}",
                f"图像判读重点：{visual}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "当前为 text-only 设置，不提供图片。若选项涉及具体局部形态，只能基于题面有限信息作保守判断。",
                "",
            ]
        )
    lines.append("请从以下 4 个选项中选择唯一最佳答案。")
    return {
        "question": "\n".join(lines),
        "modalities": ["text", "image"] if mode == "multimodal" else ["text"],
        **({"image": record["input"]["multimodal"]["image"]} if mode == "multimodal" else {}),
    }


def balance_choices(record: dict[str, Any]) -> list[dict[str, str]]:
    choices = [dict(choice) for choice in record["choices"]]
    longest_index = int(hashlib.sha256(record["id"].encode("utf-8")).hexdigest()[:8], 16) % len(choices)
    for idx, choice in enumerate(choices):
        text = choice["text"]
        if len(text) < 72:
            clause = PADDING_CLAUSES[(idx + len(record["id"])) % len(PADDING_CLAUSES)]
            if not text.endswith("。"):
                text += "。"
            text += clause
        if idx == longest_index:
            if not text.endswith("。"):
                text += "。"
            text += LONG_NEUTRAL_SUFFIXES[(idx + len(record["id"])) % len(LONG_NEUTRAL_SUFFIXES)]
        choice["text"] = text
    return choices


def rebuild_record(record: dict[str, Any]) -> dict[str, Any]:
    new_record = dict(record)
    choices = balance_choices(record)
    choice_text = "\n".join(f"{c['id']}. {c['text']}" for c in choices)
    new_record["choices"] = choices
    new_record["input"] = {
        "text_only": make_prompt(record, "text_only"),
        "multimodal": make_prompt(record, "multimodal"),
    }
    new_record["input"]["text_only"]["question"] += "\n" + choice_text
    new_record["input"]["multimodal"]["question"] += "\n" + choice_text
    new_record["benchmark_version"] = "v2_less_leaky_prompt_balanced_choices"
    new_record["audit_notes"] = {
        "answer_fields_kept_for_manual_check": True,
        "planned_final_schema": "After manual QA, split model-facing input.jsonl and answer_key.jsonl.",
        "text_prompt_policy": "No chart-derived periods, anomaly locations, change points, lag relations, or interval verdicts in text-only prompt.",
    }
    new_record["correct_choice"] = next(c for c in choices if c["id"] == record["answer_key"])
    return new_record


def summarize(records: list[dict[str, Any]], output_dir: Path) -> None:
    answer_counts = Counter(r["answer_key"] for r in records)
    task_counts = Counter(r["task_type"] for r in records)
    difficulty_counts = Counter(str(r.get("difficulty_level")) for r in records)
    longest_hits = sum(max(r["choices"], key=lambda c: len(c["text"]))["id"] == r["answer_key"] for r in records)
    shortest_hits = sum(min(r["choices"], key=lambda c: len(c["text"]))["id"] == r["answer_key"] for r in records)
    same_except_image = 0
    for r in records:
        t = r["input"]["text_only"]["question"]
        m = r["input"]["multimodal"]["question"]
        m = re.sub(r"\n请同时参考图片：images/ts_agent_\d+\.png\n图像判读重点：[^\n]+\n", "\n", m)
        m = m.replace("当前为 text-only 设置，不提供图片。若选项涉及具体局部形态，只能基于题面有限信息作保守判断。\n", "")
        same_except_image += t == m

    correct_lengths = [len(r["correct_choice"]["text"]) for r in records]
    distractor_lengths = [
        len(c["text"])
        for r in records
        for c in r["choices"]
        if c["id"] != r["answer_key"]
    ]
    summary = {
        "num_samples": len(records),
        "dataset": "ts_agent_mcq_v2.jsonl",
        "source_dataset": "dataset_benchmark_mcq_with_series/ts_agent_mcq.jsonl",
        "image_dir": "images",
        "answer_key_counts": dict(answer_counts),
        "task_type_counts": dict(task_counts),
        "difficulty_counts": dict(difficulty_counts),
        "sanity_checks": {
            "choose_longest_accuracy": round(longest_hits / len(records), 4),
            "choose_shortest_accuracy": round(shortest_hits / len(records), 4),
            "avg_correct_choice_chars": round(mean(correct_lengths), 2),
            "avg_distractor_choice_chars": round(mean(distractor_lengths), 2),
            "text_and_multimodal_identical_after_image_line_removed": same_except_image,
        },
        "notes": [
            "Answer-side fields are intentionally retained for manual inspection in this draft.",
            "Model-facing prompt fields avoid explicit chart-derived evidence in text-only mode.",
            "A final release should split input.jsonl and answer_key.jsonl after QA.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 选择题版时序建模 Agent Benchmark v2 摘要",
        "",
        f"- 样本数：{len(records)}",
        "- 问答文件：`ts_agent_mcq_v2.jsonl`",
        "- 图片目录：`images/`",
        "- 答案字段：本版仍保留，方便人工检查；最终版建议拆分 input/answer_key。",
        f"- 选最长项准确率：{longest_hits / len(records):.2%}",
        f"- 选最短项准确率：{shortest_hits / len(records):.2%}",
        "",
        "## 答案位置分布",
        "",
        "| 选项 | 数量 |",
        "| --- | ---: |",
    ]
    for key in CHOICE_IDS:
        lines.append(f"| {key} | {answer_counts.get(key, 0)} |")
    lines.extend(["", "## 主要改动", "", "- text-only 题面不再泄露周期、异常位置、突变点、滞后关系等图表解析结论。", "- multimodal 题面增加图像判读重点，明确需要从图片读取证据。", "- 选项长度做平衡，降低最长项捷径。"])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(source_dir / "ts_agent_mcq.jsonl")
    rebuilt = [rebuild_record(r) for r in records]
    for src in (source_dir / "images").glob("*.png"):
        dst = image_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    write_jsonl(output_dir / "ts_agent_mcq_v2.jsonl", rebuilt)
    summarize(rebuilt, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("dataset_benchmark_mcq_with_series"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_benchmark_mcq_with_series_v2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(args.source_dir, args.output_dir)
    print(f"Wrote v2 MCQ dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
