from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .common import done_ids, iter_jsonl
from .compact_json import append_jsonl, json_safe, strict_dumps
from .trajectories import (
    _candidate_tiebreak_key,
    _generate_candidate,
    _runtime_task,
    _runtime_user_content,
    _sanitize_tool_result,
    _select_with_judge,
)
from .questions import QUESTION_RUNTIME_FORMAT_VERSION
from .trajectory_verify import deterministic_trajectory_review


TRAJECTORY_FORMAT_VERSION = "tool_trajectory_v2.2_compact"
COMPETITION_AUDIT_FORMAT_VERSION = "trajectory_competition_audit_v2.1_compact"


def _bounded_value(value: Any, depth: int = 0) -> Any:
    """Keep result evidence readable without embedding unbounded arrays or runtime noise."""

    value = json_safe(_sanitize_tool_result(value))
    if depth >= 6:
        return {"truncated": True, "reason": "maximum_depth"}
    if isinstance(value, str):
        return value if len(value) <= 2000 else value[:2000] + "…"
    if isinstance(value, list):
        items = [_bounded_value(item, depth + 1) for item in value[:20]]
        if len(value) > 20:
            items.append({"truncated": True, "omitted_items": len(value) - 20})
        return items
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        visible_items = [(key, child) for key, child in value.items() if key not in {"session_id", "created_at"}]
        for key, child in visible_items[:40]:
            result[str(key)] = _bounded_value(child, depth + 1)
        if len(visible_items) > 40:
            result["truncated"] = True
            result["omitted_fields"] = len(visible_items) - 40
        return result
    return value


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for source in messages:
        message = json_safe(source)
        if message.get("role") == "tool" and isinstance(message.get("content"), str):
            try:
                content = json.loads(message["content"])
            except json.JSONDecodeError:
                pass
            else:
                message["content"] = strict_dumps(content)
        normalized.append(message)
    return normalized


def _decision_by_call_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        decision = str(message.get("content") or "").strip()
        for call in message.get("tool_calls") or []:
            call_id = str(call.get("id") or "")
            if call_id and decision:
                decisions[call_id] = decision
    return decisions


def _result_summary(event: dict[str, Any]) -> Any:
    result = event.get("result")
    if not isinstance(result, dict):
        return _bounded_value(result)
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        if structured.get("summary") is not None:
            return _bounded_value(structured["summary"])
        if structured.get("error") is not None:
            return _bounded_value({"error": structured["error"]})
    content = result.get("content")
    if isinstance(content, list):
        text = [item.get("text") for item in content if isinstance(item, dict) and item.get("text")]
        if text:
            return _bounded_value({"text": text})
    return _bounded_value(result)


def _returned_model_names(event: dict[str, Any]) -> list[str]:
    if event.get("name") != "model_catalog_search" or not event.get("ok"):
        return []
    result = event.get("result") or {}
    structured = result.get("structuredContent") if isinstance(result, dict) else {}
    summary = structured.get("summary") if isinstance(structured, dict) else {}
    names = []
    for candidate in summary.get("candidates", []) if isinstance(summary, dict) else []:
        if isinstance(candidate, dict) and candidate.get("name"):
            names.append(str(candidate["name"]))
    return names


def _tool_call_views(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages") or []
    decisions = _decision_by_call_id(messages)
    result: list[dict[str, Any]] = []
    for event in record.get("tool_events") or []:
        call_id = str(event.get("tool_call_id") or "")
        item: dict[str, Any] = {
            "tool_call_id": call_id,
            "name": str(event.get("name") or ""),
            "arguments": json_safe(event.get("arguments") or {}),
            "ok": bool(event.get("ok")),
            "result": _result_summary(event),
        }
        if decisions.get(call_id):
            item["decision"] = decisions[call_id]
        if not item["ok"]:
            item["error"] = {
                "type": str(event.get("error_type") or "tool_error"),
                "message": str(event.get("error") or "Tool call failed"),
            }
        artifacts = []
        for artifact in event.get("created_artifacts") or []:
            if not isinstance(artifact, dict):
                continue
            artifacts.append({
                key: json_safe(artifact.get(key))
                for key in ["artifact_id", "kind", "uri", "description"]
                if artifact.get(key) is not None
            })
        if artifacts:
            item["artifacts"] = artifacts
        images = [Path(str(path)).name for path in event.get("image_paths") or []]
        if images:
            item["images"] = images
        returned_names = _returned_model_names(event)
        if returned_names:
            item["returned_model_names"] = returned_names
        result.append(item)
    return result


def _metrics(record: dict[str, Any]) -> dict[str, Any]:
    events = record.get("tool_events") or []
    successful = [event for event in events if event.get("ok")]
    usage = record.get("usage") or {}
    return {
        "tool_calls": len(events),
        "successful_tool_calls": len(successful),
        "failed_tool_calls": len(events) - len(successful),
        "distinct_successful_tools": sorted({str(event.get("name")) for event in successful}),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "latency_seconds": round(float(record.get("latency_seconds", 0) or 0), 3),
        "generation_attempts": int(record.get("generation_attempt", 1) or 1),
    }


def _final_answer(record: dict[str, Any]) -> str:
    answer = str(record.get("final_answer") or "").strip()
    if answer:
        return answer
    for message in reversed(record.get("messages") or []):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            return str(message.get("content") or "").strip()
    return ""


def _question_view(row: dict[str, Any], candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    messages = (candidate or {}).get("messages") or []
    if len(messages) > 1 and messages[1].get("role") == "user":
        text = str(messages[1].get("content") or "")
    else:
        text = _runtime_user_content(row, "uploads/dataset.csv")
    task = _runtime_task(row)
    result = {
        "text": text,
        "task": task.get("category"),
        "input_mode": task.get("input_mode"),
        "model_catalog_scope": task.get("model_catalog_scope", "none"),
    }
    return {key: json_safe(value) for key, value in result.items() if value is not None}


def _review_view(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(review.get("passed")),
        "score": float(review.get("score", 0) or 0),
        "flags": list(review.get("flags") or []),
        "warnings": list(review.get("warnings") or []),
        "observed": json_safe(review.get("observed") or {}),
    }


def _attempt_view(attempt: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {
        "attempt": int(attempt.get("attempt", 0) or 0),
        "error": {
            "type": str(attempt.get("error_type") or attempt.get("terminal_error_type") or "generation_error"),
            "message": str(attempt.get("error") or attempt.get("terminal_error") or "Candidate generation failed"),
        },
        "tool_calls": _tool_call_views(attempt),
        "metrics": _metrics(attempt),
    }
    return item


def _candidate_audit_view(candidate: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    failed_attempts = [_attempt_view(attempt) for attempt in candidate.get("failed_attempts") or []]
    item: dict[str, Any] = {
        "candidate_index": int(candidate.get("candidate_index", 0) or 0),
        "model": candidate.get("model"),
        "generation_status": "ok" if candidate.get("status") == "ok" else "error",
        "tool_calls": _tool_call_views(candidate) if candidate.get("status") == "ok" else [],
        "final_answer": _final_answer(candidate),
        "metrics": _metrics(candidate),
        "review": _review_view(review),
        "failed_attempts": failed_attempts,
    }
    if candidate.get("status") != "ok":
        item["error"] = {
            "type": str(candidate.get("error_type") or candidate.get("terminal_error_type") or "generation_error"),
            "message": str(candidate.get("error") or candidate.get("terminal_error") or "Candidate generation failed"),
        }
    return json_safe(item)


def _selector_view(selector: dict[str, Any] | None) -> dict[str, Any] | None:
    if selector is None:
        return None
    result: dict[str, Any] = {
        "status": selector.get("status"),
        "model": selector.get("model"),
        "decision_rule": selector.get("decision_rule"),
    }
    response = selector.get("response")
    if isinstance(response, dict):
        result["candidate_scores"] = json_safe(response.get("candidate_scores") or [])
        result["winner"] = response.get("winner")
        result["rationale"] = response.get("rationale")
    if selector.get("error"):
        result["error"] = {
            "type": selector.get("error_type"),
            "message": selector.get("error"),
        }
    return {key: value for key, value in result.items() if value is not None}


def _selected_record(
    row: dict[str, Any],
    candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    winner_index: int,
    selection_rule: str,
    selector: dict[str, Any] | None,
) -> dict[str, Any]:
    return json_safe({
        "id": row["id"],
        "format_version": TRAJECTORY_FORMAT_VERSION,
        "question": _question_view(row, candidate),
        "generation_status": "ok",
        "model": candidate.get("model"),
        "candidate_index": winner_index,
        "messages": _normalize_messages(candidate.get("messages") or []),
        "tools": json_safe(candidate.get("tools") or []),
        "tool_calls": _tool_call_views(candidate),
        "final_answer": _final_answer(candidate),
        "metrics": _metrics(candidate),
        "competition": {
            "selection_rule": selection_rule,
            "winner_index": winner_index,
            "candidate_models": [item.get("model") for item in candidates],
            "review_scores": [float(review.get("score", 0) or 0) for review in reviews],
            "selector_model": selector.get("model") if selector else None,
            "audit_id": row["id"],
        },
    })


def _failed_record(row: dict[str, Any], selection_rule: str) -> dict[str, Any]:
    return {
        "id": row["id"],
        "format_version": TRAJECTORY_FORMAT_VERSION,
        "question": _question_view(row),
        "generation_status": "error",
        "error": {
            "type": "both_candidates_failed",
            "message": "Both trajectory candidates failed deterministic validation.",
        },
        "competition": {
            "selection_rule": selection_rule,
            "winner_index": None,
            "audit_id": row["id"],
        },
    }


def _audit_record(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    selection_rule: str,
    winner_index: int | None,
    selector: dict[str, Any] | None,
) -> dict[str, Any]:
    reference_candidate = next((candidate for candidate in candidates if candidate.get("messages")), None)
    return json_safe({
        "id": row["id"],
        "format_version": COMPETITION_AUDIT_FORMAT_VERSION,
        "question": _question_view(row, reference_candidate),
        "candidates": [
            _candidate_audit_view(candidate, review)
            for candidate, review in zip(candidates, reviews)
        ],
        "selection": {
            "rule": selection_rule,
            "winner_index": winner_index,
            "selector": _selector_view(selector),
        },
    })


def generate_trajectories(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
    candidate_configs: list[dict[str, Any]],
    selector_config: dict[str, Any],
    model_catalog: list[dict[str, Any]],
    repo_root: Path,
    workspace_root: Path,
    resume: bool = False,
    limit: int | None = None,
    max_attempts: int = 2,
    min_tool_calls: int = 1,
    min_distinct_tools: int = 1,
    max_tool_calls: int = 8,
    max_tool_result_chars: int = 16000,
) -> None:
    if len(candidate_configs) != 2:
        raise ValueError("models.trajectory_candidates must contain exactly two tool-call-capable models")
    identities = {(str(config.get("base_url", "")), str(config.get("model", ""))) for config in candidate_configs}
    if len(identities) != 2 or any(not base_url or not model for base_url, model in identities):
        raise ValueError("trajectory_candidates must configure two distinct provider/model identities")
    if not resume:
        for path in [output_path, audit_path]:
            if path.exists():
                path.unlink()
    completed = done_ids(output_path) if resume else set()
    rows = list(iter_jsonl(input_path))
    rows = rows[:limit] if limit else rows
    stale = [row.get("id") for row in rows if row.get("format_version") != QUESTION_RUNTIME_FORMAT_VERSION]
    if stale:
        raise RuntimeError(f"Compact trajectories require {QUESTION_RUNTIME_FORMAT_VERSION}: {', '.join(map(str, stale[:3]))}")
    for row in rows:
        if row["id"] in completed:
            continue
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    _generate_candidate,
                    row,
                    dict(config),
                    index,
                    repo_root,
                    workspace_root,
                    model_catalog,
                    max_attempts,
                    min_tool_calls,
                    min_distinct_tools,
                    max_tool_calls,
                    max_tool_result_chars,
                )
                for index, config in enumerate(candidate_configs)
            ]
            candidates = [future.result() for future in futures]
        reviews = [deterministic_trajectory_review(candidate, model_catalog=model_catalog) for candidate in candidates]
        passed = [index for index, review in enumerate(reviews) if review["passed"]]
        selector_audit: dict[str, Any] | None = None
        winner_index: int | None = None
        if len(passed) == 1:
            winner_index = passed[0]
            selection_rule = "single_hard_gate_pass"
        elif len(passed) == 2:
            selection_rule = "multimodal_judge"
            try:
                winner_index, selector_audit = _select_with_judge(row, candidates, reviews, selector_config)
            except Exception as exc:  # noqa: BLE001
                winner_index = int(min(candidates, key=_candidate_tiebreak_key)["candidate_index"])
                selection_rule = "selector_error_deterministic_fallback"
                selector_audit = {
                    "status": "error",
                    "model": selector_config.get("model"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        else:
            selection_rule = "both_hard_gate_failed"
        append_jsonl(
            audit_path,
            _audit_record(row, candidates, reviews, selection_rule, winner_index, selector_audit),
        )
        if winner_index is None:
            output = _failed_record(row, selection_rule)
        else:
            output = _selected_record(
                row,
                candidates[winner_index],
                candidates,
                reviews,
                winner_index,
                selection_rule,
                selector_audit,
            )
        append_jsonl(output_path, output)
