from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .common import append_jsonl, done_ids, iter_jsonl, stable_hash
from .trajectories import MODEL_CATALOG_SEARCH_TOOL, TRAJECTORY_FORMAT_VERSION, _sanitize_tool_result, _scope_models


FORBIDDEN_TRAINING_KEYS = ["truth_path", "ground_truth", "anomaly_indices", "hidden_truth_for_verification_only"]
FORBIDDEN_PROMPT_HINTS = ["prepared_channels", "primary_tools", "model_route", "matched_subclass", "routing_rule"]
GENERIC_MODEL_METHODS = {
    "transformer",
    "lstm",
    "rnn",
    "cnn",
    "arima",
    "sarima",
    "ets",
    "holt-winters",
    "holt_winters",
    "prophet",
    "seasonalnaive",
    "seasonal_naive",
    "linear regression",
    "deep learning",
    "foundation model",
    "线性回归",
    "深度学习",
    "基础模型",
}


def _normalized_model_term(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _is_generic_model_method(value: str) -> bool:
    return _normalized_model_term(value) in GENERIC_MODEL_METHODS


def _catalog_mentions(text: str, models: list[dict[str, Any]]) -> set[str]:
    mentions: set[str] = set()
    for model in models:
        name = str(model.get("name", "")).strip()
        if _is_generic_model_method(name):
            continue
        if len(name) >= 3 and re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text, flags=re.I):
            mentions.add(name)
    return mentions


def _generic_method_mentions(text: str) -> set[str]:
    mentions: set[str] = set()
    for method in GENERIC_MODEL_METHODS:
        if len(method) < 3:
            continue
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(method)}(?![A-Za-z0-9_])", text, flags=re.I):
            mentions.add(method)
    return mentions


def _returned_model_names(events: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for event in events:
        if event.get("name") != MODEL_CATALOG_SEARCH_TOOL or not event.get("ok"):
            continue
        structured = (event.get("result") or {}).get("structuredContent", {})
        summary = structured.get("summary", {}) if isinstance(structured, dict) else {}
        for candidate in summary.get("candidates", []) if isinstance(summary, dict) else []:
            if isinstance(candidate, dict) and candidate.get("name"):
                names.add(str(candidate["name"]))
    return names


def _event_evidence_hash(event: dict[str, Any]) -> str:
    visible = {
        "ok": bool(event.get("ok")),
        "tool_response": _sanitize_tool_result(event.get("result")),
        "created_artifacts": [
            {key: artifact.get(key) for key in ["artifact_id", "kind", "uri", "description", "metadata"]}
            for artifact in event.get("created_artifacts", [])
        ],
    }
    if event.get("trajectory_budget") is not None:
        visible["trajectory_budget"] = event["trajectory_budget"]
    return stable_hash(visible)


def deterministic_trajectory_review(record: dict[str, Any], model_catalog: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    flags: list[str] = []
    warnings: list[str] = []
    if record.get("status") != "ok":
        return {"passed": False, "score": 0.0, "flags": ["generation_error"], "warnings": warnings}
    if record.get("format_version") != TRAJECTORY_FORMAT_VERSION:
        flags.append("unsupported_format_version")
    row = record.get("question_record") or {}
    messages = record.get("messages") or []
    tools = record.get("tools") or []
    events = record.get("tool_events") or []
    definitions = {tool.get("function", {}).get("name"): tool for tool in tools if tool.get("function", {}).get("name")}
    candidate_names = {str(name) for name in row.get("allowed_tools", []) if str(name)}
    runtime_task = row.get("task") if isinstance(row.get("task"), dict) else {}
    resources = row.get("resources") if isinstance(row.get("resources"), dict) else {}
    scope = str(runtime_task.get("model_catalog_scope", "none"))
    allowed_names = candidate_names | ({MODEL_CATALOG_SEARCH_TOOL} if scope != "none" else set())
    if set(definitions) - allowed_names:
        flags.append("tool_definition_outside_allowlist")
    if scope == "none" and MODEL_CATALOG_SEARCH_TOOL in definitions:
        flags.append("unexpected_model_catalog_definition")
    if scope != "none" and MODEL_CATALOG_SEARCH_TOOL not in definitions:
        flags.append("missing_model_catalog_definition")
    structure_ok = len(messages) >= 5 and messages[0].get("role") == "system" and messages[1].get("role") == "user"
    if not structure_ok:
        flags.append("invalid_message_prefix")
    call_ids: set[str] = set()
    ordered_calls: list[tuple[str, str, dict[str, Any]]] = []
    expected_tool_id: str | None = None
    event_by_id = {str(event.get("tool_call_id")): event for event in events}
    for index, message in enumerate(messages[2:], 2):
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            calls = message["tool_calls"]
            if expected_tool_id is not None or not isinstance(calls, list) or len(calls) != 1:
                flags.append(f"invalid_assistant_tool_turn:{index}")
                continue
            call = calls[0]
            call_id = str(call.get("id", ""))
            function = call.get("function") or {}
            name = str(function.get("name", ""))
            if not call_id or call_id in call_ids:
                flags.append(f"invalid_or_duplicate_call_id:{index}")
            call_ids.add(call_id)
            expected_tool_id = call_id
            if name not in definitions:
                flags.append(f"undefined_tool:{name}")
                arguments = {}
            else:
                try:
                    arguments = json.loads(function.get("arguments", "{}"))
                    validation_errors = list(Draft202012Validator(definitions[name]["function"]["parameters"]).iter_errors(arguments))
                    event = event_by_id.get(call_id, {})
                    if validation_errors and event.get("ok"):
                        flags.append(f"successful_invalid_tool_arguments:{name}")
                    elif validation_errors:
                        warnings.append(f"recovered_invalid_tool_arguments:{name}")
                except Exception as exc:  # noqa: BLE001
                    flags.append(f"malformed_tool_arguments:{name}:{type(exc).__name__}")
                    arguments = {}
            ordered_calls.append((call_id, name, arguments))
        elif role == "tool":
            if expected_tool_id is None or message.get("tool_call_id") != expected_tool_id:
                flags.append(f"orphan_or_mismatched_tool_result:{index}")
            if not isinstance(message.get("content"), str) or not message.get("content"):
                flags.append(f"empty_tool_result:{index}")
            expected_tool_id = None
        elif role == "assistant":
            if expected_tool_id is not None or index != len(messages) - 1:
                flags.append(f"misplaced_final_answer:{index}")
            if not str(message.get("content", "")).strip():
                flags.append("empty_final_answer")
        else:
            flags.append(f"unexpected_role:{role}:{index}")
    if expected_tool_id is not None:
        flags.append("missing_tool_result")
    if not messages or messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls"):
        flags.append("trajectory_does_not_end_with_final_assistant")
    event_ids = [str(event.get("tool_call_id")) for event in events]
    ordered_ids = [call_id for call_id, _, _ in ordered_calls]
    if event_ids != ordered_ids:
        flags.append("tool_event_call_id_mismatch")
    for (call_id, name, arguments), event in zip(ordered_calls, events):
        if event.get("name") != name or event.get("arguments") != arguments:
            flags.append(f"tool_event_payload_mismatch:{call_id}")
        expected_hash = event.get("result_audit", {}).get("full_result_sha256")
        if expected_hash and expected_hash != _event_evidence_hash(event):
            flags.append(f"tool_event_evidence_hash_mismatch:{call_id}")
    successful_events = [event for event in events if event.get("ok")]
    successful = len(successful_events)
    distinct = {str(event.get("name")) for event in successful_events}
    if successful < 1:
        flags.append("no_successful_tool_call")
    if int(record.get("successful_tool_calls", -1)) != successful:
        flags.append("successful_tool_count_mismatch")
    if set(record.get("distinct_successful_tools", [])) != distinct:
        flags.append("distinct_successful_tools_mismatch")
    signatures: set[str] = set()
    for event in successful_events:
        signature = stable_hash({"name": event.get("name"), "arguments": event.get("arguments")})
        if signature in signatures:
            warnings.append(f"redundant_identical_successful_call:{event.get('name')}")
        signatures.add(signature)
    failed_count = len(events) - successful
    if failed_count:
        warnings.append(f"recovered_tool_failures:{failed_count}")
    primary: set[str] = set()
    final_answer = str(record.get("final_answer", ""))
    if len(final_answer.strip()) < 60:
        warnings.append("final_answer_brief")
    original_path = str((resources.get("dataset") or {}).get("path", ""))
    training_projection = [
        {key: message.get(key) for key in ["role", "content", "tool_calls", "tool_call_id", "name"] if key in message}
        for message in messages
    ]
    training_text = json.dumps(training_projection, ensure_ascii=False)
    if original_path and original_path in training_text:
        flags.append("absolute_dataset_path_leak")
    if any(prefix in training_text for prefix in ["/Users/", "/private/tmp/", "/private/var/", "C:\\"]):
        flags.append("runtime_absolute_path_leak")
    for forbidden in FORBIDDEN_TRAINING_KEYS:
        if forbidden in training_text:
            flags.append(f"hidden_key_leak:{forbidden}")
    system_text = str(messages[0].get("content", "")) if messages else ""
    for forbidden in FORBIDDEN_PROMPT_HINTS:
        if forbidden in system_text:
            flags.append(f"prompt_route_leak:{forbidden}")
    if runtime_task.get("input_mode") == "image_text" and (len(messages) < 2 or not messages[1].get("images")):
        flags.append("missing_initial_image")

    returned_names = _returned_model_names(events)
    generic_mentions = _generic_method_mentions(final_answer)
    if scope == "none" and generic_mentions:
        warnings.append("generic_model_method_mentioned_without_catalog:" + ",".join(sorted(generic_mentions)))
    if model_catalog:
        scoped = _scope_models(model_catalog, scope) if scope != "none" else []
        scoped_names = {str(model.get("name")) for model in scoped}
        if not returned_names <= scoped_names:
            flags.append("model_catalog_scope_violation")
        mentioned = _catalog_mentions(final_answer, model_catalog)
        unauthorized = mentioned - returned_names
        if unauthorized:
            flags.append("unqueried_model_name:" + ",".join(sorted(unauthorized)))
    elif scope == "none" and returned_names:
        flags.append("unexpected_model_catalog_result")

    structure_score = 1.0 if not any(flag.startswith(("invalid_", "orphan_", "missing_tool_result", "misplaced_", "trajectory_", "tool_event_")) for flag in flags) else 0.0
    evidence_score = 1.0 if not any(flag.startswith(("absolute_", "runtime_absolute_", "hidden_key_", "model_catalog_", "unqueried_")) for flag in flags) else 0.0
    tool_score = 1.0 if successful >= 1 and not any(flag.startswith(("undefined_", "malformed_", "successful_invalid_")) for flag in flags) else 0.0
    relevance_score = 1.0 if not primary or bool(primary & distinct) else 0.7
    answer_score = 1.0 if final_answer.strip() else 0.0
    score = round(0.30 * structure_score + 0.30 * evidence_score + 0.20 * tool_score + 0.10 * relevance_score + 0.10 * answer_score, 4)
    fatal_prefixes = (
        "generation_error", "unsupported_", "invalid_", "empty_", "undefined_", "orphan_", "missing_tool_result",
        "trajectory_", "tool_event_", "no_successful_", "successful_tool_", "distinct_successful_", "malformed_",
        "successful_invalid_", "absolute_", "runtime_absolute_", "hidden_key_", "prompt_route_", "missing_initial_",
        "unexpected_model_", "missing_model_", "model_catalog_", "unqueried_",
    )
    fatal = any(flag.startswith(fatal_prefixes) for flag in flags)
    return {
        "passed": score >= 0.8 and not fatal,
        "score": score,
        "flags": flags,
        "warnings": warnings,
        "subscores": {"structure": structure_score, "evidence": evidence_score, "tools": tool_score, "soft_relevance": relevance_score, "answer": answer_score},
        "observed": {"successful_calls": successful, "failed_calls": failed_count, "returned_model_names": sorted(returned_names)},
    }


def verify_trajectories(
    input_path: Path,
    verified_path: Path,
    rejected_path: Path,
    resume: bool = False,
    limit: int | None = None,
    model_catalog: list[dict[str, Any]] | None = None,
) -> None:
    if not resume:
        for path in [verified_path, rejected_path]:
            if path.exists():
                path.unlink()
    completed = (done_ids(verified_path) | done_ids(rejected_path)) if resume else set()
    rows = list(iter_jsonl(input_path))
    rows = rows[:limit] if limit else rows
    for record in rows:
        if record["id"] in completed:
            continue
        review = deterministic_trajectory_review(record, model_catalog=model_catalog)
        output = {**record, "trajectory_review": review, "status": "verified" if review["passed"] else "rejected"}
        append_jsonl(verified_path if review["passed"] else rejected_path, output)
