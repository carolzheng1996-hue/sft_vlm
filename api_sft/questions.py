from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from .api_client import OpenAICompatibleClient, parse_json_object
from .common import append_jsonl, iter_jsonl, load_yaml, read_json, stable_hash, write_json, write_jsonl
from .signal_generator import GENERATOR_VERSION


QUESTION_SPEC_VERSION = "4.0"
QUESTION_WRITER_PROMPT_ID = "question_writer_v2"
QUESTION_CONTRACT_VERSION = "2"
TOOL_EXECUTION_SYSTEM_PROMPT_ID = "tsa_tool_execution_v2"
QUESTION_MESSAGE_FORMAT = "neutral_local_images_v1"
COVERAGE_REPORT_NAME = "coverage_report.json"
TOOL_EXECUTION_SYSTEM_PROMPT = (Path(__file__).with_name("prompts") / "tool_execution_system.txt").read_text(encoding="utf-8").strip()

TOOL_FOCUS = {
    "data_profile": ["data_profile", "data_quality_check", "series_overview", "summary_stats", "shape_distri", "Trend_linear", "adf_test", "seasonality_detector", "FFT", "STL", "cpt_detector", "plot"],
    "similarity_analysis": ["series_overview", "summary_stats", "pear_cross_correlation", "spear_cross_correlation", "dtw_distance", "plot"],
    "model_result_analysis": ["summary_stats", "compute_acf", "white_noise_test", "seasonality_detector", "cpt_detector", "plot"],
    "model_selection": ["data_profile", "data_quality_check", "series_overview", "adf_test", "seasonality_detector", "feature_generate", "plot"],
    "tool_use": ["data_scan", "data_profile", "data_quality_check", "series_overview", "channel_stats", "channel_values", "plot", "ts_imputation", "cpt_detector"],
}

TASK_LABELS = {
    "data_profile": "数据画像",
    "similarity_analysis": "序列相似性分析",
    "model_result_analysis": "模型结果分析",
    "model_selection": "时序模型选择",
    "tool_use": "时序分析工具使用",
}

VISUAL_REASON = {
    "low": "结构化字段和统计摘要是当前判断的主要证据；只有出现待定位的时间结构时才需要补图。",
    "medium": "文本可完成初筛，针对性图像可能帮助确认局部形态、序列差异或训练动态。",
    "high": "时间位置、局部形态或跨序列关系难以由汇总统计充分表达，图像通常有较高信息增益。",
}

PLOT_PRIORITY = {
    "data_profile": ["overview.png", "anomaly_changepoint.png", "acf_fft.png", "stl.png", "missingness_heatmap.png"],
    "similarity_analysis": ["small_multiples.png", "normalized_overlay.png", "correlation_heatmap.png", "dtw_alignment.png"],
    "model_result_analysis": ["residual_diagnostics.png", "training_curves.png", "monitoring_drift.png"],
    "model_selection": ["overview.png", "small_multiples.png", "acf_fft.png", "stl.png"],
    "tool_use": ["missingness_heatmap.png", "overview.png", "distribution.png"],
}

MODEL_CATALOG_REQUIRED = {
    "similarity_global_local_model",
    "result_model_tradeoff",
}
MODEL_CATALOG_CONDITIONAL = {
    "result_training_dynamics",
    "result_data_concept_drift",
    "result_retrain_rollback_review",
}
EXPLICIT_TOOL_MENTION_TASKS = {
    "tool_profile_only",
    "tool_stationarity_period_order",
    "tool_similarity_choice",
    "tool_conflicting_results",
    "tool_hard_negative",
}

ANSWER_BLUEPRINT_PATTERNS = [
    r"答案必须覆盖",
    r"请按照以下(?:步骤|流程)",
    r"工具调用顺序",
    r"逐步写明",
    r"文本证据.*图像证据",
    r"待验证假设",
    r"Top\s*-?\s*3",
    r"rolling\s*/\s*expanding",
    r"不选其他模型的原因",
    r"请以.{0,12}(?:方式|结构)组织答案",
]
IMAGE_CLAIM_PATTERNS = [r"图中(?:显示|可见)", r"图片中(?:显示|可见)", r"附图(?:显示|表明)", r"请看图", r"结合所给图像"]


def _choose_images(scenario: dict[str, Any], limit: int = 3) -> list[str]:
    by_name = {Path(path).name: path for path in scenario.get("images", [])}
    priority = ["missingness_heatmap.png","overview.png","correlation_heatmap.png"] if scenario.get("pattern") == "missing_blocks" else PLOT_PRIORITY[scenario["task"]]
    chosen = [by_name[name] for name in priority if name in by_name]
    return (chosen or scenario.get("images", []))[:limit]


def _image_attachments(images: list[str]) -> list[dict[str, str]]:
    return [
        {
            "path": path,
            "filename": Path(path).name,
            "media_type": "image/png",
            "source_type": "local_path",
        }
        for path in images
    ]


def _question_messages(question: str, images: list[str], system_prompt: str) -> list[dict[str, str]]:
    image_prefix = "".join("<image>\n" for _ in images)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": image_prefix + question},
    ]


UNIVERSAL_INGEST_TOOLS = ["data_profile", "data_quality_check", "data_convert", "list_channels"]
TOOL_POOL_SIZE = 20
TOOL_POOL_EXCLUSIONS = {
    "knowledge_search",
    "model_cancel",
    "model_list_jobs",
    "model_status",
    "model_train",
    "model_training_preflight",
}


def _candidate_tool_pool(
    task_spec: dict[str, Any],
    task: str,
    tool_map: dict[str, dict[str, Any]],
    seed: int,
    question_id: str,
    target_size: int = TOOL_POOL_SIZE,
) -> list[dict[str, Any]]:
    required = list(dict.fromkeys(task_spec["preferred_tools"] + UNIVERSAL_INGEST_TOOLS))
    focused = list(dict.fromkeys(required + TOOL_FOCUS[task]))
    selected = [name for name in focused if name in tool_map and name not in TOOL_POOL_EXCLUSIONS]
    if len(selected) > target_size:
        protected = [name for name in required if name in selected]
        selected = list(dict.fromkeys(protected + [name for name in selected if name not in protected]))[:target_size]
    distractors = sorted(
        (
            name for name in tool_map
            if name not in selected and name not in TOOL_POOL_EXCLUSIONS
        ),
        key=lambda name: stable_hash(f"{seed}:{question_id}:{name}"),
    )
    selected.extend(distractors[: max(0, target_size - len(selected))])
    return [tool_map[name] for name in selected]


def _spec_compatible(spec: dict[str, Any], scenario: dict[str, Any]) -> bool:
    sid = spec["id"]
    pattern = scenario.get("pattern", "")
    rules = {
        "profile_missingness_patterns": {"missing_blocks"},
        "profile_anomaly_types": {"anomaly_change"},
        "profile_change_variance_drift": {"anomaly_change"},
        "profile_intermittent_longtail": {"intermittent"},
        "profile_scale_variance_heterogeneity": {"multi_period", "missing_blocks"},
        "similarity_phase_dtw": {"phase_groups"},
        "similarity_lag_lead": {"phase_groups"},
        "similarity_distribution_not_shape": {"distribution_mismatch"},
        "similarity_anomalous_series": {"sequence_anomaly"},
        "similarity_motif_local": {"motif"},
        "result_training_dynamics": {"overfit", "underfit"},
        "result_data_concept_drift": {"drift"},
        "result_retrain_rollback_review": {"drift"},
        "selection_uni_multivariate": {"covariates"},
        "selection_future_covariates": {"covariates"},
        "selection_intermittent": {"intermittent"},
        "selection_probabilistic_interval": {"interval"},
        "selection_anomaly_task": {"sequence_anomaly"},
        "selection_hierarchy_reconciliation": {"hierarchical"},
        "tool_profile_only": {"stationary", "leakage"},
        "tool_plot_required": {"missing_blocks", "anomaly_change"},
        "tool_stationarity_period_order": {"trend_period", "multi_period"},
        "tool_similarity_choice": {"scaled_groups"},
        "tool_missing_imputation": {"missing_blocks"},
        "tool_anomaly_change_order": {"anomaly_change"},
        "tool_normalize_before_similarity": {"scaled_groups"},
        "tool_image_needs_statistics": {"trend_period", "anomaly_change"},
        "tool_avoid_unnecessary_plot": {"stationary", "leakage"},
        "tool_conflicting_results": {"multi_period"},
        "tool_hard_negative": {"leakage"},
    }
    if sid in rules:
        return pattern in rules[sid]
    if spec["parent_task"] == "model_result_analysis" and sid not in MODEL_CATALOG_CONDITIONAL:
        return pattern in {"residual_period", "residual_hetero"}
    return True


def assign_task_specs(scenarios: list[dict[str, Any]], task_specs: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    assignments: dict[str, dict[str, Any]] = {}
    for task in TASK_LABELS:
        task_scenarios = [scenario for scenario in scenarios if scenario["task"] == task]
        specs = [spec for spec in task_specs if spec["parent_task"] == task]
        unused = list(task_scenarios)
        if not specs:
            raise ValueError(f"No task-pool entries for {task}")
        for spec in specs:
            match = next((scenario for scenario in unused if _spec_compatible(spec, scenario)), None)
            if match is not None:
                assignments[match["id"]] = spec
                unused.remove(match)
        offset = 0
        for scenario in unused:
            compatible = [spec for spec in specs if _spec_compatible(spec, scenario)] or specs
            assignments[scenario["id"]] = compatible[offset % len(compatible)]
            offset += 1
    return [(scenario, assignments[scenario["id"]]) for scenario in scenarios if scenario["id"] in assignments]


def _weighted_modality(policy: dict[str, float], key: str) -> str:
    value = int(stable_hash(key)[:12], 16) / float(16**12)
    cumulative = 0.0
    for name in ["text_only", "image_text", "paired"]:
        cumulative += float(policy[name])
        if value < cumulative:
            return name
    return "paired"


def _stratified_categories(count: int, policy: dict[str, float], key: str) -> list[str]:
    names = ["text_only", "image_text", "paired"]
    raw = {name: count * float(policy[name]) for name in names}
    quotas = {name: int(raw[name]) for name in names}
    remainder = count - sum(quotas.values())
    order = sorted(names, key=lambda item: (raw[item] - quotas[item], stable_hash(f"{key}:{item}")), reverse=True)
    for name in order[:remainder]:
        quotas[name] += 1
    categories = [name for name in names for _ in range(quotas[name])]
    indexed = sorted(enumerate(categories), key=lambda item: stable_hash(f"{key}:{item[0]}:{item[1]}"))
    return [name for _, name in indexed]


def plan_modality_sampling(assignments: list[tuple[dict[str, Any], dict[str, Any]]], config: dict[str, Any], seed: int) -> list[tuple[dict[str, Any], dict[str, Any], list[str], str]]:
    policies = config["policies"]
    for value in policies.values():
        if set(value) != {"text_only", "image_text", "paired"} or abs(sum(float(x) for x in value.values()) - 1) > 1e-6:
            raise ValueError("Each modality policy must contain text_only/image_text/paired and sum to 1")
    if not config.get("ensure_each_subtask_both_modalities", False):
        planned: list[list[Any]] = []
        levels: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for scenario, spec in assignments:
            image_policy = spec.get("image_policy", "conditional")
            level = "low" if image_policy == "low" else "high" if image_policy == "high" else "medium"
            levels[level].append((scenario, spec))
        for level, items in levels.items():
            ordered = sorted(items, key=lambda item: stable_hash(f"{seed}:{level}:{item[1]['id']}:{item[0]['id']}"))
            categories = _stratified_categories(len(ordered), policies[level], f"{seed}:{level}:categories")
            for (scenario, spec), choice in zip(ordered, categories):
                modes = ["text_only", "image_text"] if choice == "paired" else [choice]
                planned.append([scenario, spec, modes, f"stratified_{level}_{choice}"])
        if config.get("ensure_each_parent_task_both_modalities", True):
            for task in TASK_LABELS:
                indices = [i for i, item in enumerate(planned) if item[0]["task"] == task]
                observed = {mode for i in indices for mode in planned[i][2]}
                if "text_only" not in observed and indices:
                    planned[indices[0]][2] = ["text_only", "image_text"]
                    planned[indices[0]][3] = "forced_parent_task_pair"
                if "image_text" not in observed and indices:
                    planned[indices[-1]][2] = ["text_only", "image_text"]
                    planned[indices[-1]][3] = "forced_parent_task_pair"
        return [(scenario, spec, modes, reason) for scenario, spec, modes, reason in planned]
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for item in assignments:
        groups[item[1]["id"]].append(item)
    plans: dict[str, tuple[list[str], str]] = {}
    for subtask, items in groups.items():
        ordered = sorted(items, key=lambda item: stable_hash(f"{seed}:{subtask}:{item[0]['id']}"))
        image_policy = ordered[0][1].get("image_policy", "conditional")
        level = "low" if image_policy == "low" else "high" if image_policy == "high" else "medium"
        primary = "text_only" if level == "low" else "image_text"
        counter = "image_text" if primary == "text_only" else "text_only"
        if len(ordered) == 1:
            plans[ordered[0][0]["id"]] = (["text_only", "image_text"], "forced_pair_for_subtask_coverage")
            continue
        if level == "medium":
            plans[ordered[0][0]["id"]] = (["text_only", "image_text"], "medium_value_paired_anchor")
            plans[ordered[1][0]["id"]] = ([counter], "subtask_counter_modality")
            start = 2
        else:
            plans[ordered[0][0]["id"]] = ([primary], "policy_primary_modality")
            plans[ordered[1][0]["id"]] = ([counter], "subtask_counter_modality")
            start = 2
        for scenario, _ in ordered[start:]:
            choice = _weighted_modality(policies[level], f"{seed}:{subtask}:{scenario['id']}:modality")
            modes = ["text_only", "image_text"] if choice == "paired" else [choice]
            plans[scenario["id"]] = (modes, f"stratified_{level}_{choice}")
    return [(scenario, spec, *plans[scenario["id"]]) for scenario, spec in assignments]


def _time_range(data_path: str | None) -> dict[str, str] | None:
    if not data_path or not Path(data_path).exists():
        return None
    values: list[str] = []
    with Path(data_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("time") not in {None, ""}:
                values.append(str(row["time"]))
    if not values:
        return None
    try:
        numeric = [float(value) for value in values]
        return {"start": str(min(numeric)).removesuffix(".0"), "end": str(max(numeric)).removesuffix(".0")}
    except ValueError:
        return {"start": min(values), "end": max(values)}


def build_evidence_packet(scenario: dict[str, Any]) -> dict[str, Any]:
    visible = scenario["visible_context"]
    packet: dict[str, Any] = {
        "schema": {"columns": visible.get("columns", [])},
        "data_layout": visible.get("data_layout"),
        "wide_column_map": visible.get("wide_column_map", {}),
        "data_scale": {
            "row_count": visible.get("row_count"),
            "series_count": visible.get("series_count"),
            "history_length_per_series": visible.get("history_length_per_series"),
        },
        "time_index": {"frequency": visible.get("frequency"), "observed_range": _time_range(scenario.get("data_path"))},
        "statistics": visible.get("summary", {}),
        "business_constraints": visible.get("business_constraints", {}),
    }
    for key in ["known_future_covariates", "unknown_future_covariates", "target_zero_ratio", "hierarchy", "forecast_requirement"]:
        if key in visible:
            packet[key] = visible[key]
    return packet


def _model_catalog_scope(spec: dict[str, Any], task_goal: str) -> str:
    relevant = (
        spec["parent_task"] == "model_selection"
        or spec["id"] in MODEL_CATALOG_REQUIRED
        or spec["id"] in MODEL_CATALOG_CONDITIONAL
    )
    if not relevant:
        return "none"
    return "anomaly_detection" if "anomaly" in task_goal else "forecast"


def _image_supervision(spec: dict[str, Any]) -> tuple[str, str]:
    policy = spec.get("image_policy", "conditional")
    if policy == "low":
        return "text_only", "low"
    if policy == "high":
        return "image_text", "high"
    return "text_then_image", "medium"


def _task_specs(path: Path) -> list[dict[str, Any]]:
    pool = load_yaml(path)
    policy = pool.get("task_difficulty", {})
    hard = set(policy.get("hard", [])); default = str(policy.get("default", "medium"))
    result = []
    for source in pool["tasks"]:
        spec = dict(source); spec["task_difficulty"] = str(source.get("task_difficulty", "hard" if source["id"] in hard else default)); result.append(spec)
    return result


def build_question_specs(
    scenarios_path: Path,
    tools_path: Path,
    models_path: Path,
    output_path: Path,
    questions_per_scenario: int = 2,
    limit: int | None = None,
    resume: bool = False,
    task_pool_path: Path | None = None,
    modality_sampling: dict[str, Any] | None = None,
    seed: int = 20260712,
) -> list[dict[str, Any]]:
    scenarios = list(iter_jsonl(scenarios_path))
    if limit is not None:
        scenarios = scenarios[:limit]
    stale = [scenario.get("id") for scenario in scenarios if scenario.get("generator_version") != GENERATOR_VERSION]
    if stale:
        raise RuntimeError(f"QuestionSpec requires generator {GENERATOR_VERSION}; regenerate stale scenarios first: {', '.join(str(item) for item in stale[:3])}")
    tools = read_json(tools_path)["tools"]
    # The QuestionSpec records only a coarse catalog scope. Model entries stay
    # in the runtime catalog and are never frozen into the question prompt.
    read_json(models_path)
    tool_map = {tool["name"]: tool for tool in tools}
    task_pool_path = task_pool_path or Path(__file__).with_name("task_pool.yaml")
    task_specs = _task_specs(task_pool_path)
    assignments = assign_task_specs(scenarios, task_specs)
    if modality_sampling and modality_sampling.get("enabled", True):
        planned = plan_modality_sampling(assignments, modality_sampling, seed)
    else:
        planned = [(scenario, spec, ["text_only", "image_text"][:questions_per_scenario], "legacy_all_paired") for scenario, spec in assignments]
    existing = {row["id"]: row for row in iter_jsonl(output_path)} if resume and output_path.exists() else {}
    rows: list[dict[str, Any]] = []
    for scenario, task_spec, modes, sampling_reason in planned:
        recommended_mode, image_value = _image_supervision(task_spec)
        analysis_difficulty = str(scenario.get("analysis_difficulty", scenario.get("difficulty", "medium")))
        task_difficulty = str(task_spec.get("task_difficulty", "medium"))
        question_difficulty = "hard" if analysis_difficulty == "hard" and task_difficulty == "hard" else "medium"
        pair_id = f"pair_{scenario['id']}_{task_spec['id']}" if len(modes) == 2 else None
        evidence_packet = build_evidence_packet(scenario)
        for index, mode in enumerate(modes):
            question_id = f"{scenario['id']}_q{index + 1:02d}_{mode}"
            candidate_tools = _candidate_tool_pool(task_spec, scenario["task"], tool_map, seed, question_id)
            images = _choose_images(scenario) if mode == "image_text" else []
            allowed_mentions = task_spec["preferred_tools"] if task_spec["id"] in EXPLICIT_TOOL_MENTION_TASKS else []
            record = {
                "id": question_id,
                "question_spec_version": QUESTION_SPEC_VERSION,
                "question_group_id": pair_id or question_id,
                "scenario_id": scenario["id"],
                "pair_id": pair_id,
                "is_paired": bool(pair_id),
                "pair_role": mode if pair_id else f"standalone_{mode}",
                "split_group": scenario["id"],
                "modality_sampling_reason": sampling_reason,
                "task": scenario["task"],
                "task_label": TASK_LABELS[scenario["task"]],
                "subtask_id": task_spec["id"],
                "subtask_title": task_spec["title"],
                "task_goal": scenario["task_goal"],
                "series_count": scenario["series_count"],
                "history_length": scenario["history_length"],
                "signal_complexity": scenario.get("signal_complexity"),
                "analysis_difficulty": analysis_difficulty,
                "task_difficulty": task_difficulty,
                "difficulty": question_difficulty,
                "input_mode": mode,
                "data_layout": scenario.get("data_layout"),
                "wide_column_map": scenario.get("wide_column_map", {}),
                "data_path": scenario["data_path"],
                "dataset_attachment": {
                    "path": scenario["data_path"],
                    "format": "csv",
                    "source_type": "local_path",
                },
                "images": images,
                "image_inventory": [Path(path).name for path in images],
                "image_attachments": _image_attachments(images),
                "evidence_packet": evidence_packet,
                "visible_context": scenario["visible_context"],
                "recommended_mode": recommended_mode,
                "image_value": image_value,
                "visual_reason": VISUAL_REASON[image_value],
                "recommended_plots": [] if image_value == "low" else [Path(path).name for path in _choose_images(scenario)],
                "text_can_answer": ["schema、规模和摘要中的显式统计", "业务约束和工具前置条件"],
                "image_should_answer": ["时间位置和局部形态", "跨序列形状与相位关系"] if image_value != "low" else [],
                "requires_statistical_confirmation": ["周期与平稳性", "异常与变点显著性", "相关而非因果"],
                "model_catalog_scope": _model_catalog_scope(task_spec, scenario["task_goal"]),
                "system_prompt_id": TOOL_EXECUTION_SYSTEM_PROMPT_ID,
                "system_prompt": TOOL_EXECUTION_SYSTEM_PROMPT,
                "message_format": QUESTION_MESSAGE_FORMAT,
                "trajectory_requirement": "tool_execution",
                "candidate_tools": candidate_tools,
                "primary_tools": task_spec["preferred_tools"],
                "required_answer_elements": task_spec["required_elements"],
                "internal_rubric": {
                    "task_instruction": task_spec["instruction"],
                    "required_elements": task_spec["required_elements"],
                    "preferred_tools": task_spec["preferred_tools"],
                    "image_policy": task_spec.get("image_policy", "conditional"),
                    "expected_decision_points": 3 if question_difficulty == "hard" else 2,
                    "allowed_user_tool_mentions": allowed_mentions,
                    "evidence_boundaries": ["可见摘要不是原始数据", "图像观察不能替代统计检验", "未知结果不得写成事实"],
                },
                "scenario_hash": scenario["scenario_hash"],
            }
            record["question_spec_hash"] = stable_hash(record)
            previous = existing.get(question_id)
            rows.append(previous if previous and previous.get("question_spec_hash") == record["question_spec_hash"] else record)
    rows.sort(key=lambda row: row["id"])
    write_jsonl(output_path, rows)
    return rows


def question_writer_payload(specs: list[dict[str, Any]], feedback: list[str] | None = None) -> dict[str, Any]:
    first = specs[0]
    image_inventory = sorted({name for spec in specs for name in spec.get("image_inventory", [])})
    model_scope = str(first.get("model_catalog_scope", "none"))
    existing_result_comparison = model_scope == "none" and first.get("task") == "model_result_analysis"
    payload: dict[str, Any] = {
        "task": {"category": first["task_label"], "fine_grained_task": first["subtask_title"], "business_goal": first["task_goal"]},
        "scenario": {"series_count": first["series_count"], "history_length": first["history_length"]},
        "difficulty": first["difficulty"],
        "input_modes": sorted({spec["input_mode"] for spec in specs}),
        "visible_evidence": first["evidence_packet"],
        "available_image_types": image_inventory,
        "decision_boundaries": {
            "model_selection": (
                "existing labeled model/result comparison is allowed, but proposing or naming new model families is forbidden"
                if existing_result_comparison
                else "forbidden: keep the question on data checks, conversion, diagnosis, or evidence gaps"
                if model_scope == "none"
                else f"allowed only within the coarse {model_scope} task class; do not name catalog models"
            ),
        },
        "required_output": {
            "user_request": "自然的中文业务问题；不要重复结构化材料中的精确数值",
            "decision_points": "问题实际包含的两个或三个决策点，仅用于审计，不会放入用户消息",
            "constraint_key": "实际使用的 business_constraints 字段名",
            "facts_used": "引用的 visible_evidence 字段路径列表",
        },
    }
    if feedback:
        payload["validation_feedback"] = feedback
    return payload


def _context_block(packet: dict[str, Any], data_path: str) -> str:
    return (
        "可访问的数据资源：\n"
        f"- dataset_path: {data_path}\n\n"
        "已提供的结构化材料（仅代表当前可见信息）：\n"
        + json.dumps(packet, ensure_ascii=False, indent=2)
    )


def materialize_question_record(
    spec: dict[str, Any],
    generation: dict[str, Any],
    quality: dict[str, Any],
    attempts: list[dict[str, Any]] | None = None,
    shared_generation_with: str | None = None,
) -> dict[str, Any]:
    """Build a directly consumable system+user Question record without calling an API."""
    out = dict(spec)
    context_block = _context_block(spec["evidence_packet"], spec["data_path"])
    question = generation["user_request"] + "\n\n" + context_block
    images = list(spec.get("images", []))
    system_prompt = str(spec.get("system_prompt") or TOOL_EXECUTION_SYSTEM_PROMPT)
    out.update({
        "user_request": generation["user_request"],
        "context_block": context_block,
        "question": question,
        "messages": _question_messages(question, images, system_prompt),
        "message_format": QUESTION_MESSAGE_FORMAT,
        "dataset_attachment": spec.get("dataset_attachment") or {
            "path": spec["data_path"], "format": "csv", "source_type": "local_path",
        },
        "image_attachments": _image_attachments(images),
        "question_quality": quality,
        "question_contract_version": QUESTION_CONTRACT_VERSION,
        "question_source": "llm_direct_from_question_spec",
        "question_writer_prompt_id": QUESTION_WRITER_PROMPT_ID,
        "question_generation": {
            "model": generation.get("model"),
            "attempts": attempts or [],
            "shared_generation_with": shared_generation_with,
            "decision_points": generation.get("decision_points", []),
            "constraint_key": generation.get("constraint_key", ""),
            "facts_used": generation.get("facts_used", []),
        },
    })
    return out


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?%?", text))


def _known_constraint_keys(spec: dict[str, Any]) -> set[str]:
    constraints = spec.get("evidence_packet", {}).get("business_constraints", {})
    return {str(key) for key in constraints}


MODEL_DECISION_PATTERNS = [
    r"模型选型|候选模型|推荐模型|模型排行|模型排序",
    r"(?:选择|推荐|比较|评估|确定|决定|部署|替换).{0,16}(?:预测方法|预测模型|异常检测模型|模型|算法)",
    r"(?:预测方法|预测模型|异常检测模型|模型|算法).{0,16}(?:选择|推荐|比较|选型|排序|部署|替换)",
    r"(?:哪种|什么|哪个).{0,8}(?:预测方法|预测模型|异常检测模型|模型|算法)",
    r"用什么.{0,6}(?:方法|模型|算法).{0,6}(?:预测|检测)",
]


def _requests_model_decision(text: str) -> bool:
    """Detect requests to choose a new model family, not comparisons of supplied A/B results."""

    anonymous_redacted = re.sub(r"模型\s*(?:[A-Z]|[甲乙丙丁]|[一二三四]|\d+)", "已有方案", text, flags=re.I)
    compact = re.sub(r"\s+", "", anonymous_redacted)
    return any(re.search(pattern, compact, flags=re.I) for pattern in MODEL_DECISION_PATTERNS)


def _evidence_paths(value: Any, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths.update(_evidence_paths(child, path))
    elif isinstance(value, list):
        paths.add(prefix)
    return paths


def validate_question(
    text: str,
    spec: dict[str, Any] | None = None,
    writer_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    writer_metadata = writer_metadata or {}
    decision_points = writer_metadata.get("decision_points", [])
    if not isinstance(decision_points, list):
        decision_points = []
    constraint_key = str(writer_metadata.get("constraint_key", ""))
    facts_used = writer_metadata.get("facts_used", [])
    if not isinstance(facts_used, list):
        facts_used = []
    minimum_decisions = int(spec.get("internal_rubric", {}).get("expected_decision_points", 2)) if spec else 2
    allowed_numbers = _numbers(json.dumps(spec.get("evidence_packet", {}), ensure_ascii=False)) if spec else set()
    unexpected_numbers = sorted(_numbers(text) - allowed_numbers) if spec else sorted(_numbers(text))
    blueprint_hits = [pattern for pattern in ANSWER_BLUEPRINT_PATTERNS if re.search(pattern, text, flags=re.I | re.S)]
    image_claim_hits = [pattern for pattern in IMAGE_CLAIM_PATTERNS if re.search(pattern, text, flags=re.I)]
    allowed_tools = set(spec.get("internal_rubric", {}).get("allowed_user_tool_mentions", [])) if spec else set()
    named_tools: set[str] = set()
    if spec:
        for tool in spec.get("candidate_tools", []):
            name = str(tool.get("name", ""))
            if name and re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text, flags=re.I):
                named_tools.add(name)
    unexpected_tools = sorted(named_tools - allowed_tools)
    named_models: list[str] = []
    no_model_catalog = bool(spec and spec.get("model_catalog_scope") == "none")
    model_decision_requested = _requests_model_decision(text)
    known_paths = _evidence_paths(spec.get("evidence_packet", {})) if spec else set()
    known_paths |= {f"visible_evidence.{path}" for path in known_paths}
    unknown_fact_paths = sorted({str(path) for path in facts_used if str(path) not in known_paths}) if spec else []
    checks = {
        "minimum_length": len(text.strip()) >= 60,
        "maximum_length": len(text.strip()) <= 360,
        "multiple_decisions": len([point for point in decision_points if str(point).strip()]) >= minimum_decisions if spec else len(text.strip()) >= 60,
        "business_constraint": constraint_key in _known_constraint_keys(spec) if spec else bool(re.search(r"成本|时延|解释|算力|风险|业务", text)),
        "no_answer_blueprint": not blueprint_hits,
        "grounded_numbers": not unexpected_numbers,
        "grounded_fact_paths": bool(facts_used) and not unknown_fact_paths if spec else True,
        "modality_neutral": not image_claim_hits,
        "allowed_tool_mentions": not unexpected_tools,
        "no_model_name_leak": not named_models,
        "model_scope": not (no_model_catalog and model_decision_requested),
        "not_single_fact": not bool(re.fullmatch(r".{0,20}(有没有趋势|哪个点是异常|使用DTW吗|推荐哪个模型).{0,10}", text.strip())),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "character_count": len(text.strip()),
        "complexity_score": round(sum(checks.values()) / len(checks), 4),
        "details": {
            "blueprint_hits": blueprint_hits,
            "unexpected_numbers": unexpected_numbers,
            "unknown_fact_paths": unknown_fact_paths,
            "image_claim_hits": image_claim_hits,
            "unexpected_tool_mentions": unexpected_tools,
            "model_name_leaks": sorted(set(named_models)),
            "model_decision_requested": model_decision_requested,
            "decision_point_count": len(decision_points),
            "constraint_key": constraint_key,
        },
    }


def _validation_feedback(quality: dict[str, Any]) -> list[str]:
    messages = {
        "minimum_length": "问题过短，需要形成至少两个相互关联的业务决策。",
        "maximum_length": "问题过长，删除答案提纲和通用执行要求。",
        "multiple_decisions": "决策点数量不足。",
        "business_constraint": "constraint_key 必须取自已提供的 business_constraints。",
        "no_answer_blueprint": "删除答案章节、工具顺序、Top-3 或回测模板等解题提示。",
        "grounded_numbers": "删除结构化材料中不存在的精确数字。",
        "grounded_fact_paths": "facts_used 必须至少包含一个真实存在的 visible_evidence 字段路径。",
        "modality_neutral": "不要声称图片显示了什么，也不要直接提示看图。",
        "allowed_tool_mentions": "除允许的工具边界题外，不要点名具体工具。",
        "no_model_name_leak": "不要提前给出任何具体模型名称。",
        "model_scope": "该子任务不允许模型选型；请改为数据检查、转换、诊断或证据补充决策。",
        "not_single_fact": "问题不能退化为单一事实判断。",
    }
    return [messages[name] for name, passed in quality["checks"].items() if not passed]


def _groups(rows: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("question_group_id") or row["id"])].append(row)
    return [grouped[key] for key in sorted(grouped)]


def generate_questions(
    specs_path: Path,
    output_path: Path,
    rejected_path: Path,
    model_config: dict[str, Any],
    writer_prompt_path: Path,
    resume: bool = False,
    limit: int | None = None,
    max_attempts: int = 2,
) -> list[dict[str, Any]]:
    if not resume:
        if output_path.exists():
            output_path.unlink()
        if rejected_path.exists():
            rejected_path.unlink()
    specs = list(iter_jsonl(specs_path))
    spec_by_id = {row["id"]: row for row in specs}
    spec_hashes = {row["id"]:row.get("question_spec_hash") for row in specs}
    all_existing = {row["id"]: row for row in iter_jsonl(output_path)} if output_path.exists() else {}
    existing: dict[str, dict[str, Any]] = {}
    invalid_existing_ids: set[str] = set()
    for row_id, row in all_existing.items():
        spec = spec_by_id.get(row_id)
        metadata = row.get("question_generation", {})
        quality = validate_question(str(row.get("user_request", "")), spec, metadata) if spec else {"passed": False}
        reusable = bool(
            spec
            and row.get("question_spec_hash")
            and row.get("question_spec_hash") == spec_hashes.get(row_id)
            and quality["passed"]
        )
        if reusable:
            upgraded = dict(row)
            upgraded["question_quality"] = quality
            upgraded["question_contract_version"] = QUESTION_CONTRACT_VERSION
            existing[row_id] = upgraded
        else:
            invalid_existing_ids.add(row_id)
    groups = _groups(specs)
    if limit is not None:
        selected = groups[:limit]
        selected_keys = {str(group[0].get("question_group_id") or group[0]["id"]) for group in selected}
        repair_groups = [
            group for group in groups[limit:]
            if any(spec["id"] in invalid_existing_ids for spec in group)
            and str(group[0].get("question_group_id") or group[0]["id"]) not in selected_keys
        ]
        groups = selected + repair_groups
    writer_system = writer_prompt_path.read_text(encoding="utf-8").strip()
    client = OpenAICompatibleClient(model_config)
    for group in groups:
        missing = [spec for spec in group if spec["id"] not in existing]
        if not missing:
            continue
        reusable = next((existing[spec["id"]] for spec in group if spec["id"] in existing), None)
        if reusable is None:
            reusable = None
        generation: dict[str, Any] | None = None
        quality: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        if reusable:
            generation = {
                "user_request": reusable["user_request"],
                "decision_points": reusable.get("question_generation", {}).get("decision_points", []),
                "constraint_key": reusable.get("question_generation", {}).get("constraint_key", ""),
                "facts_used": reusable.get("question_generation", {}).get("facts_used", []),
                "model": reusable.get("question_generation", {}).get("model"),
            }
            quality = validate_question(generation["user_request"], group[0], generation)
        else:
            feedback: list[str] = []
            for attempt in range(1, max_attempts + 1):
                payload = question_writer_payload(group, feedback)
                try:
                    raw, usage, latency = client.complete([
                        {"role": "system", "content": writer_system},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ])
                    obj = parse_json_object(raw)
                    candidate = {
                        "user_request": str(obj.get("user_request", "")).strip(),
                        "decision_points": obj.get("decision_points", []),
                        "constraint_key": str(obj.get("constraint_key", "")).strip(),
                        "facts_used": obj.get("facts_used", []),
                        "model": model_config.get("model"),
                    }
                    quality = validate_question(candidate["user_request"], group[0], candidate)
                    attempts.append({"attempt": attempt, "status": "accepted" if quality["passed"] else "validation_failed", "usage": usage, "latency_seconds": round(latency, 3), "quality": quality})
                    if quality["passed"]:
                        generation = candidate
                        break
                    feedback = _validation_feedback(quality)
                except Exception as exc:
                    feedback = [f"上一次输出无法解析或请求失败：{type(exc).__name__}。请严格输出所需 JSON 字段。"]
                    attempts.append({"attempt": attempt, "status": "error", "error": str(exc), "usage": {}, "latency_seconds": None})
        if generation is None or quality is None or not quality["passed"]:
            for spec in missing:
                append_jsonl(rejected_path, {"id": spec["id"], "stage": "question_generation", "reason": "writer_failed_after_retries", "attempts": attempts, "question_spec": spec})
            continue
        for index, spec in enumerate(missing):
            out = materialize_question_record(
                spec,
                generation,
                quality,
                attempts if index == 0 else [],
                None if index == 0 else missing[0]["id"],
            )
            append_jsonl(output_path, out)
            existing[out["id"]] = out
    rows = [existing[key] for key in sorted(existing)]
    write_jsonl(output_path,rows)
    if rejected_path.exists():
        unresolved = [row for row in iter_jsonl(rejected_path) if row.get("id") not in existing]
        if unresolved:
            write_jsonl(rejected_path, unresolved)
        else:
            rejected_path.unlink()
    task_specs = _task_specs(Path(__file__).with_name("task_pool.yaml"))
    all_tools = {name for spec in task_specs for name in spec["preferred_tools"]}
    write_coverage_report(rows, output_path.with_name(COVERAGE_REPORT_NAME), task_specs, all_tools, False)
    return rows


def rewrite_questions(
    input_path: Path,
    output_path: Path,
    rejected_path: Path,
    model_config: dict[str, Any],
    writer_prompt_path: Path,
    resume: bool = False,
    limit: int | None = None,
    max_attempts: int = 2,
) -> list[dict[str, Any]]:
    """Deprecated compatibility alias for the v2 direct question writer."""
    return generate_questions(input_path, output_path, rejected_path, model_config, writer_prompt_path, resume, limit, max_attempts)


def refresh_question_records(specs_path: Path, output_path: Path) -> list[dict[str, Any]]:
    """Refresh the deterministic message envelope of accepted questions without an LLM call."""
    specs = {row["id"]: row for row in iter_jsonl(specs_path)}
    refreshed: list[dict[str, Any]] = []
    for old in iter_jsonl(output_path):
        spec = specs.get(old.get("id"))
        if not spec:
            continue
        if old.get("scenario_hash") != spec.get("scenario_hash") or old.get("subtask_id") != spec.get("subtask_id"):
            continue
        metadata = old.get("question_generation", {})
        generation = {
            "user_request": old.get("user_request", ""),
            "decision_points": metadata.get("decision_points", []),
            "constraint_key": metadata.get("constraint_key", ""),
            "facts_used": metadata.get("facts_used", []),
            "model": metadata.get("model"),
        }
        if not generation["user_request"]:
            continue
        quality = validate_question(generation["user_request"], spec, generation)
        if not quality["passed"]:
            continue
        refreshed.append(materialize_question_record(
            spec,
            generation,
            quality,
            metadata.get("attempts", []),
            metadata.get("shared_generation_with"),
        ))
    refreshed.sort(key=lambda row: row["id"])
    write_jsonl(output_path, refreshed)
    task_specs = _task_specs(Path(__file__).with_name("task_pool.yaml"))
    all_tools = {name for spec in task_specs for name in spec["preferred_tools"]}
    write_coverage_report(refreshed, output_path.with_name(COVERAGE_REPORT_NAME), task_specs, all_tools, False)
    return refreshed


def write_coverage_report(
    rows: list[dict[str, Any]],
    path: Path,
    task_specs: list[dict[str, Any]] | None = None,
    all_tools: set[str] | None = None,
    require_subtask_modalities: bool = True,
) -> dict[str, Any]:
    axes = ["task", "subtask_id", "series_count", "history_length", "task_goal", "input_mode", "recommended_mode", "image_value", "difficulty"]
    rows = [{**{"subtask_id": "unspecified"}, **row} for row in rows]
    report: dict[str, Any] = {"record_count": len(rows), "axes": {axis: dict(Counter(str(row.get(axis, "missing")) for row in rows)) for axis in axes}}
    report["task_series_length"] = {}
    report["task_input_mode"] = {}
    report["subtask_input_mode"] = {}
    for row in rows:
        a = f"{row['task']}|{row['series_count']}|{row['history_length']}"
        b = f"{row['task']}|{row['input_mode']}"
        c = f"{row['subtask_id']}|{row['input_mode']}"
        report["task_series_length"][a] = report["task_series_length"].get(a, 0) + 1
        report["task_input_mode"][b] = report["task_input_mode"].get(b, 0) + 1
        report["subtask_input_mode"][c] = report["subtask_input_mode"].get(c, 0) + 1
    representatives: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        representatives.setdefault(str(row.get("question_group_id") or row.get("pair_id") or row.get("id") or f"row_{index}"), row)
    normalized = [(group, re.sub(r"\s+", "", row.get("user_request", row.get("question", "")))) for group, row in representatives.items()]
    exact_duplicates = len(normalized) - len({text for _, text in normalized})
    near = 0
    grouped_texts: dict[tuple[str, str], list[str]] = defaultdict(list)
    for _, row in representatives.items():
        grouped_texts[(row["subtask_id"], row["input_mode"])].append(re.sub(r"\s+", "", row.get("user_request", row.get("question", ""))))
    for texts in grouped_texts.values():
        for i in range(len(texts)):
            if any(SequenceMatcher(None, texts[i], texts[j]).ratio() > 0.96 for j in range(i)):
                near += 1
    average_chars = round(sum(len(row.get("user_request", row.get("question", ""))) for row in rows) / max(1, len(rows)), 2)
    average_complexity = round(sum(row.get("question_quality", {}).get("complexity_score", 0) for row in rows) / max(1, len(rows)), 4)
    report["quality"] = {"exact_duplicate_count": exact_duplicates, "near_duplicate_count": near, "average_user_request_characters": average_chars, "average_complexity_score": average_complexity}
    scenario_modes: dict[str, set[str]] = defaultdict(set)
    scenario_values: dict[str, str] = {}
    for row in rows:
        sid = str(row.get("scenario_id", row.get("id")))
        scenario_modes[sid].add(row["input_mode"])
        scenario_values[sid] = row["image_value"]
    categories = {sid: ("paired" if modes == {"text_only", "image_text"} else next(iter(modes))) for sid, modes in scenario_modes.items()}
    by_value = {level: dict(Counter(category for sid, category in categories.items() if scenario_values[sid] == level)) for level in ["low", "medium", "high"]}
    report["modality_sampling"] = {"scenario_count": len(categories), "overall": dict(Counter(categories.values())), "by_image_value": by_value, "paired_scenario_count": sum(category == "paired" for category in categories.values())}
    expected = {
        "task": {"data_profile", "similarity_analysis", "model_result_analysis", "model_selection", "tool_use"},
        "series_count": {"single", "multiple"},
        "history_length": {"short", "long"},
        "input_mode": {"text_only", "image_text"},
        "recommended_mode": {"text_only", "text_then_image", "image_text"},
    }
    missing = {axis: sorted(values - {str(row.get(axis)) for row in rows}) for axis, values in expected.items()}
    report["missing_required_values"] = {key: value for key, value in missing.items() if value}
    report["passed"] = not report["missing_required_values"] and exact_duplicates == 0
    if task_specs is not None:
        expected_subtasks = {spec["id"] for spec in task_specs}
        observed_subtasks = {row["subtask_id"] for row in rows}
        missing_pairs = [f"{sid}|{mode}" for sid in sorted(expected_subtasks) for mode in ["text_only", "image_text"] if not any(row["subtask_id"] == sid and row["input_mode"] == mode for row in rows)]
        report["task_pool"] = {"expected_count": len(expected_subtasks), "covered_count": len(expected_subtasks & observed_subtasks), "missing_subtasks": sorted(expected_subtasks - observed_subtasks), "missing_subtask_modalities": missing_pairs}
        preferred_tools = {name for spec in task_specs for name in spec["preferred_tools"]}
        observed_primary = {name for row in rows for name in row.get("primary_tools", [])}
        expected_tools = all_tools or preferred_tools
        report["tool_pool"] = {"expected_count": len(expected_tools), "covered_count": len(expected_tools & observed_primary), "missing_tools": sorted(expected_tools - observed_primary)}
        if report["task_pool"]["missing_subtasks"] or (require_subtask_modalities and missing_pairs) or report["tool_pool"]["missing_tools"]:
            report["passed"] = False
    write_json(path, report)
    return report
