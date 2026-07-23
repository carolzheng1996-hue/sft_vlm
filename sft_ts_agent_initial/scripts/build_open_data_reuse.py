#!/usr/bin/env python3
"""Normalize reusable open-data time-series questions and derive SFT seed questions.

Inputs:
- open_data/train_cot.jsonl
- open_data/timeMQA/open_ended_QA.csv
- open_data/TSRBench/**/*.jsonl

Outputs:
- valuable original questions in a unified JSONL schema
- derived tool/model/image-routing questions in a unified JSONL schema
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALUABLE_CAPABILITIES = {
    "trend_analysis",
    "pattern_recognition",
    "seasonality_detection",
    "anomaly_detection",
    "structural_break_detection",
    "volatility_analysis",
    "statistical_profile",
    "similarity_analysis",
    "forecasting_strategy",
    "event_prediction",
    "causal_reasoning",
    "temporal_reasoning",
    "numerical_reasoning",
    "decision_strategy",
}

LOW_PRIORITY_CAPABILITIES = {
    "forecasting_strategy",
    "statistical_profile",
    "numerical_reasoning",
}


def stable_id(*parts: Any, prefix: str = "od") -> str:
    text = "::".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]}"


def compact_text(text: str, max_chars: int = 2400) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80].rstrip() + f" ... [truncated, original_length={len(text)}]"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield idx, json.loads(line)
            except json.JSONDecodeError:
                yield idx, None


def parse_time_mqa_qa(raw: str) -> dict[str, str]:
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        candidate = raw
    else:
        candidate = "{" + raw + "}"
    try:
        obj = json.loads(candidate)
        return {"question": str(obj.get("question", "")), "answer": str(obj.get("answer", ""))}
    except Exception:
        q_match = re.search(r'"question"\s*:\s*"(.*?)"\s*,\s*"answer"', raw, flags=re.S)
        a_match = re.search(r'"answer"\s*:\s*"(.*)"\s*$', raw, flags=re.S)
        return {
            "question": q_match.group(1) if q_match else raw[:1000],
            "answer": a_match.group(1) if a_match else "",
        }


def normalize_task_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def classify_capability(question: str, task_type: str = "", source_category: str = "") -> str | None:
    text = f"{question} {task_type} {source_category}".lower()
    task = normalize_task_text(task_type)
    source = normalize_task_text(source_category)
    if "forecast" in text or "predict" in text or "estimated values" in text or "prediction should cover" in text or source in {"time_series_forecasting", "event_prediction"}:
        return "event_prediction" if source == "event_prediction" else "forecasting_strategy"
    if "decision" in source or "management" in text or "treatment" in text or "trading strategies" in text or "which strategy" in text:
        return "decision_strategy"
    if "same underlying distribution" in text or "similarity" in text or "similar pattern" in text:
        return "similarity_analysis"
    if "causal" in text or "adjacency matrix" in text or source == "causal_reasoning":
        return "causal_reasoning"
    if "temporal relation" in text or source == "temporal_relation_reasoning":
        return "temporal_reasoning"
    if "deductive" in source:
        return "numerical_reasoning"
    if "numerical" in source or "calculate" in text or "mean" in text or "median" in text:
        return "numerical_reasoning" if source else "statistical_profile"
    if "structural" in text or "break" in text or "changepoint" in text or "change point" in text:
        return "structural_break_detection"
    if "anomal" in text or "outlier" in text:
        return "anomaly_detection"
    if "season" in text or "cyclic" in text or "cycle" in text or "periodic" in text or "sine" in text or "amplitude" in text:
        return "seasonality_detection"
    if "volatility" in text or "variance" in text or "noise" in text:
        return "volatility_analysis"
    if "normal distribution" in text or "statistical property" in text:
        return "statistical_profile"
    if "trend" in text or "movement" in text or "increase" in text or "decrease" in text:
        return "trend_analysis"
    if task in {"summarization", "pattern_recognition", "pattern_recognition"} or "pattern" in text:
        return "pattern_recognition"
    if task.startswith("statistical"):
        return "statistical_profile"
    return None


def extract_first_numeric_series(text: str) -> list[float] | None:
    for match in re.finditer(r"\[[^\[\]]{8,5000}\]", text):
        candidate = match.group(0)
        if not re.search(r"-?\d", candidate):
            continue
        try:
            value = ast.literal_eval(candidate)
        except Exception:
            continue
        if isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
            return [float(x) for x in value]
    return None


def looks_like_mcq(question: str, choices: list[str] | None) -> bool:
    if choices:
        return True
    return bool(re.search(r"(^|\s)(A\)|A\.|A:|\(A\)|a\))", question)) and bool(
        re.search(r"(^|\s)(B\)|B\.|B:|\(B\)|b\))", question)
    )


def compact_timeseries(ts: Any, max_len: int = 1024) -> Any:
    if ts is None:
        return None
    if not isinstance(ts, list):
        return ts
    if ts and all(isinstance(x, (int, float)) for x in ts):
        return ts[:max_len]
    compacted = []
    for series in ts[:12]:
        if isinstance(series, list):
            compacted.append(series[:max_len])
        else:
            compacted.append(series)
    return compacted


def length_of_timeseries(ts: Any) -> int | None:
    if isinstance(ts, list) and ts:
        if isinstance(ts[0], list):
            return max((len(x) for x in ts if isinstance(x, list)), default=0)
        return len(ts)
    return None


def n_series(ts: Any) -> int | None:
    if isinstance(ts, list) and ts:
        if isinstance(ts[0], list):
            return len(ts)
        return 1
    return None


def record_quality_flags(question: str, answer: str, capability: str, source: str, choices: list[str] | None = None) -> list[str]:
    flags = []
    if capability in LOW_PRIORITY_CAPABILITIES:
        flags.append("low_priority_rewrite_recommended")
    if len(question) < 40:
        flags.append("short_question")
    if len(answer) < 2 and looks_like_mcq(question, choices):
        flags.append("answer_label_only_use_choices")
    elif len(answer) < 2:
        flags.append("missing_or_short_answer")
    if len(question) > 8000:
        flags.append("very_long_question_use_excerpt_for_generation")
    question_l = question.lower()
    if source == "train_cot" and any(
        marker in question_l
        for marker in ["financial advice", "financial prediction", "stock market", "zacks", "stock price", "news published"]
    ):
        flags.append("finance_news_context_use_as_weak_signal")
    if "treatment" in question.lower() or "clinical management" in question.lower() or "cardiologist" in question.lower():
        flags.append("high_stakes_medical_use_as_tool_routing_only")
    if capability == "forecasting_strategy" and ("Predicted Prices" in answer or len(answer) > 1000):
        flags.append("do_not_sft_raw_numeric_forecast")
    return flags


def reuse_strategy(capability: str, flags: list[str]) -> str:
    if "high_stakes_medical_use_as_tool_routing_only" in flags:
        return "rewrite_to_safe_tool_routing_only"
    if "do_not_sft_raw_numeric_forecast" in flags or capability == "forecasting_strategy":
        return "rewrite_to_model_selection_and_validation"
    if capability in {"trend_analysis", "anomaly_detection", "seasonality_detection", "structural_break_detection", "similarity_analysis"}:
        return "directly_reusable_after_rewrite"
    if capability in {"decision_strategy", "causal_reasoning", "event_prediction"}:
        return "rewrite_to_decision_tool_chain"
    return "rewrite_to_expert_reasoning_seed"


def make_raw_record(
    *,
    source: str,
    source_path: Path,
    row_index: int,
    question: str,
    answer: str,
    timeseries: Any,
    choices: list[str] | None = None,
    task_type: str = "",
    question_type: str = "",
    domain: str = "",
    series_names: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    question = (question or "").strip()
    answer = (answer or "").strip()
    if not question:
        return None
    capability = classify_capability(question, task_type, source_path.stem)
    if capability not in VALUABLE_CAPABILITIES:
        return None
    flags = record_quality_flags(question, answer, capability, source, choices)
    ts = compact_timeseries(timeseries)
    rec_id = stable_id(source, source_path, row_index, question, prefix="openq")
    return {
        "id": rec_id,
        "schema_version": "open_ts_reuse_v1",
        "source": source,
        "source_path": str(source_path),
        "source_row_index": row_index,
        "original_task_type": task_type,
        "question_type": question_type,
        "normalized_capability": capability,
        "domain": domain,
        "question": question,
        "answer": answer,
        "choices": choices or [],
        "timeseries": ts,
        "timeseries_names": series_names or [],
        "timeseries_summary": {
            "num_series": n_series(ts),
            "max_length": length_of_timeseries(ts),
        },
        "reuse_strategy": reuse_strategy(capability, flags),
        "quality_flags": flags,
        "metadata": extra or {},
    }


def iter_train_cot(path: Path):
    for idx, row in iter_jsonl(path):
        if row is None:
            continue
        yield make_raw_record(
            source="train_cot",
            source_path=path,
            row_index=idx,
            question=row.get("problem", ""),
            answer=row.get("extracted_answer") or row.get("answer", ""),
            timeseries=row.get("timeseries"),
            task_type=row.get("question_type", ""),
            question_type=row.get("question_type", ""),
            extra={
                "has_cot": "<think>" in str(row.get("answer", "")),
                "ts_mean": row.get("ts_mean"),
                "ts_std": row.get("ts_std"),
                "n_target_values": row.get("n_target_values"),
            },
        )


def iter_time_mqa(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            qa = parse_time_mqa_qa(row.get("QA_list", ""))
            series = extract_first_numeric_series(qa["question"])
            yield make_raw_record(
                source="timeMQA",
                source_path=path,
                row_index=idx,
                question=qa["question"],
                answer=qa["answer"],
                timeseries=[series] if series is not None else None,
                task_type=row.get("task_type", ""),
                question_type=row.get("question_format", ""),
                domain=row.get("application_domain", ""),
                extra={"raw_task_type": row.get("task_type", "")},
            )


def iter_tsrbench(root: Path):
    for path in sorted(root.rglob("*.jsonl")):
        for idx, row in iter_jsonl(path):
            if row is None:
                continue
            mcq = row.get("multiple_choice_question") or {}
            question = row.get("question") or mcq.get("question") or ""
            answer = row.get("answer") or mcq.get("answer") or ""
            choices = row.get("choices") or mcq.get("choices") or []
            task_type = row.get("task") or path.stem
            yield make_raw_record(
                source="TSRBench",
                source_path=path,
                row_index=idx,
                question=question,
                answer=str(answer),
                timeseries=row.get("timeseries") or row.get("ts") or row.get("ts1"),
                choices=choices,
                task_type=task_type,
                question_type=row.get("question_type", "open_or_mcq"),
                domain=row.get("domain", ""),
                series_names=row.get("name_of_series", []),
                extra={
                    "dimension": path.parent.name,
                    "task_file": path.stem,
                    "type": row.get("type"),
                    "has_context": any(k in row for k in ["context", "game_info", "critical_moment"]),
                },
            )


def derived_template(record: dict[str, Any]) -> dict[str, Any]:
    cap = record["normalized_capability"]
    image_needed = cap in {
        "trend_analysis",
        "pattern_recognition",
        "seasonality_detection",
        "anomaly_detection",
        "structural_break_detection",
        "similarity_analysis",
        "volatility_analysis",
    }
    base_tools = ["profile_timeseries"]
    required = ["evidence_boundary", "tool_order_reason", "answer_not_overclaim"]
    derived_type = "tool_routing"
    question = ""
    model_focus: list[str] = []

    original_excerpt = compact_text(record["question"], max_chars=1200)

    if cap == "trend_analysis":
        tools = base_tools + ["plot_overview", "detect_anomaly_changepoint"]
        question = "请基于这道原始趋势题，把它改造成专家级时序画像任务：先判断哪些趋势结论可由数值直接支持，哪些需要 overview 图确认；再给出下一步建模或特征建议。"
        required += ["trend_direction", "local_vs_global_trend", "modeling_implication"]
    elif cap == "seasonality_detection":
        tools = base_tools + ["plot_overview", "plot_seasonal_profile"]
        question = "请把这道周期/季节性题改造成工具调用题：说明应如何用统计摘要和季节性图确认周期结构、振幅变化或伪周期，并给出适合的候选模型。"
        required += ["candidate_periods", "visual_confirmation", "seasonal_model_candidates"]
        model_focus = ["seasonal_naive", "ETS", "ARIMA/SARIMAX", "Prophet", "PatchTST/TimesNet"]
    elif cap == "anomaly_detection":
        tools = base_tools + ["plot_overview", "detect_anomaly_changepoint"]
        question = "请把这道异常识别题改造成 agent 任务：先设计异常检测和图像检查工具链，再判断异常应删除、插值、标记还是保留为业务事件。"
        required += ["anomaly_type", "image_evidence", "treatment_policy"]
    elif cap == "structural_break_detection":
        tools = base_tools + ["plot_overview", "detect_anomaly_changepoint", "drift_detection"]
        question = "请把这道结构突变题改造成诊断任务：区分点异常、短期扰动和永久 level shift，并说明验证切分、漂移监控和重训练规则如何调整。"
        required += ["changepoint_vs_spike", "validation_change", "retrain_trigger"]
    elif cap == "similarity_analysis":
        tools = base_tools + ["plot_overview", "similarity_search"]
        question = "请把这道相似性/同分布题改造成多序列分析任务：说明应比较形状、趋势、尺度、方差和分布中的哪些维度，以及是否要先标准化。"
        required += ["normalization", "distance_or_test_choice", "global_or_segment_modeling"]
    elif cap == "volatility_analysis":
        tools = base_tools + ["plot_overview", "residual_diagnostics"]
        question = "请把这道波动性题改造成建模前风险评估：说明如何区分趋势造成的表观波动、局部异方差和噪声增强，并给出模型/指标影响。"
        required += ["variance_pattern", "heteroscedasticity", "metric_implication"]
    elif cap == "forecasting_strategy":
        tools = base_tools + ["plot_overview", "rolling_backtest", "residual_diagnostics"]
        question = "请不要直接输出预测数字。请把原始预测题改造成模型选择与验证设计任务：给出 baseline、候选模型、回测方式、指标和不能过度依赖的信号。"
        required += ["baseline", "candidate_models", "rolling_origin", "bad_practices"]
        model_focus = ["seasonal_naive", "ETS", "ARIMA/SARIMAX", "LightGBM/CatBoost global forecaster", "TimesFM", "Chronos", "PatchTST/TimesNet", "TFT"]
        derived_type = "model_selection"
    elif cap == "event_prediction":
        tools = base_tools + ["plot_overview", "rolling_backtest"]
        question = "请把这道事件预测题改造成决策建模任务：说明如何从多变量时序中构造特征、选择分类/事件预测模型、设计时间切分验证和阈值策略。"
        required += ["feature_windows", "classification_vs_forecast", "threshold_policy"]
        model_focus = ["LightGBM/CatBoost global forecaster", "TFT", "PatchTST/TimesNet"]
        derived_type = "model_selection"
    elif cap in {"causal_reasoning", "temporal_reasoning"}:
        tools = base_tools + ["plot_overview", "similarity_search"]
        question = "请把这道时序因果/时序关系题改造成专家工具链任务：说明先做哪些滞后关系、同步性和可视化检查，再说明哪些结论不能仅凭相关性判断。"
        required += ["lag_reasoning", "correlation_not_causation", "validation_or_intervention"]
    elif cap == "decision_strategy":
        tools = base_tools + ["plot_overview", "rolling_backtest"]
        question = "请把这道决策题改造成安全的决策支持任务：说明应调用哪些时序分析/回测工具，如何把结果转成建议，并明确不能替代专业决策。"
        required += ["decision_criteria", "risk_boundary", "safe_response"]
        derived_type = "decision_support"
    elif cap in {"numerical_reasoning", "statistical_profile"}:
        tools = ["profile_timeseries"]
        question = "请把这道基础统计题升级成数据画像任务：不只给出单个统计量，还要说明该统计量能支持什么结论、不能支持什么结论，以及是否需要图像确认。"
        required += ["statistic_interpretation", "limitations", "next_check"]
    else:
        tools = base_tools + ["plot_overview"]
        question = "请把原题改造成专家级时序分析任务：给出工具调用顺序、证据边界和后续建模建议。"

    input_mode = "image_text" if image_needed else "text_only"
    contextual_question = (
        f"基于原始任务（来源={record['source']}，能力={cap}）：{original_excerpt}\n\n"
        f"{question}"
    )
    derived_id = stable_id(record["id"], cap, contextual_question, prefix="derivedq")
    return {
        "id": derived_id,
        "schema_version": "open_ts_derived_question_v1",
        "parent_id": record["id"],
        "source": record["source"],
        "source_path": record["source_path"],
        "normalized_capability": cap,
        "derived_task_type": derived_type,
        "input_mode": input_mode,
        "question": contextual_question,
        "rewrite_instruction": question,
        "original_question": compact_text(record["question"], max_chars=3000),
        "original_question_sha1": hashlib.sha1(record["question"].encode("utf-8")).hexdigest(),
        "original_question_length": len(record["question"]),
        "original_answer": record["answer"],
        "choices": record["choices"],
        "timeseries": record["timeseries"],
        "timeseries_names": record["timeseries_names"],
        "domain": record["domain"],
        "expected_tools": tools,
        "recommended_visual_artifacts": ["overview.png", "seasonal_profile.png", "anomaly_changepoint.png"] if image_needed else [],
        "model_focus": model_focus,
        "answer_must_cover": required,
        "reuse_strategy": record["reuse_strategy"],
        "quality_flags": record["quality_flags"],
        "metadata": {
            "original_task_type": record["original_task_type"],
            "question_type": record["question_type"],
            "timeseries_summary": record["timeseries_summary"],
            "source_metadata": record["metadata"],
        },
    }


def reservoir_by_capability(records: list[dict[str, Any]], max_per_source_capability: int, seed: int) -> list[dict[str, Any]]:
    if max_per_source_capability <= 0:
        return records
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        buckets[(rec["source"], rec["normalized_capability"])].append(rec)
    selected = []
    for key, items in sorted(buckets.items()):
        if len(items) <= max_per_source_capability:
            selected.extend(items)
        else:
            selected.extend(rng.sample(items, max_per_source_capability))
    return sorted(selected, key=lambda x: x["id"])


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_summary(path: Path, raw_records: list[dict[str, Any]], derived: list[dict[str, Any]]) -> None:
    summary = {
        "raw_records": len(raw_records),
        "derived_records": len(derived),
        "source_counts": Counter(r["source"] for r in raw_records),
        "capability_counts": Counter(r["normalized_capability"] for r in raw_records),
        "reuse_strategy_counts": Counter(r["reuse_strategy"] for r in raw_records),
        "derived_input_mode_counts": Counter(r["input_mode"] for r in derived),
        "derived_task_type_counts": Counter(r["derived_task_type"] for r in derived),
        "quality_flag_counts": Counter(flag for r in raw_records for flag in r["quality_flags"]),
    }
    serializable = json.loads(json.dumps(summary, ensure_ascii=False, default=dict))
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cot", type=Path, default=Path("open_data/train_cot.jsonl"))
    parser.add_argument("--timemqa", type=Path, default=Path("open_data/timeMQA/open_ended_QA.csv"))
    parser.add_argument("--tsrbench-root", type=Path, default=Path("open_data/TSRBench"))
    parser.add_argument("--out-raw", type=Path, default=Path("sft_ts_agent_initial/output/open_data_valuable_questions.jsonl"))
    parser.add_argument("--out-derived", type=Path, default=Path("sft_ts_agent_initial/output/open_data_derived_questions.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("sft_ts_agent_initial/output/open_data_reuse_summary.json"))
    parser.add_argument("--max-per-source-capability", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for iterator in [
        iter_train_cot(args.train_cot),
        iter_time_mqa(args.timemqa),
        iter_tsrbench(args.tsrbench_root),
    ]:
        for rec in iterator:
            if rec is not None:
                records.append(rec)

    selected = reservoir_by_capability(records, args.max_per_source_capability, args.seed)
    derived = [derived_template(rec) for rec in selected]
    write_jsonl(args.out_raw, selected)
    write_jsonl(args.out_derived, derived)
    write_summary(args.summary, selected, derived)
    print(
        json.dumps(
            {
                "out_raw": str(args.out_raw),
                "out_derived": str(args.out_derived),
                "summary": str(args.summary),
                "raw_records": len(selected),
                "derived_records": len(derived),
                "source_counts": Counter(r["source"] for r in selected),
                "capability_counts": Counter(r["normalized_capability"] for r in selected),
            },
            ensure_ascii=False,
            indent=2,
            default=dict,
        )
    )


if __name__ == "__main__":
    main()
