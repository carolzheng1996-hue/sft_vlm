from __future__ import annotations

from pathlib import Path
from typing import Any

from api_sft.common import done_ids, iter_jsonl
from api_sft.compact_json import append_jsonl
from api_sft.trajectories import TRAJECTORY_FORMAT_VERSION as FULL_TRAJECTORY_FORMAT_VERSION
from api_sft.trajectory_verify import deterministic_trajectory_review

from .trajectories import TRAJECTORY_FORMAT_VERSION


def _final_answer(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            return str(message.get("content") or "").strip()
    return ""


def _review_record(record: dict[str, Any]) -> dict[str, Any]:
    """Adapt the single-candidate compact row to the unchanged full checker."""

    tools = record.get("tools") or []
    tool_names = [
        str(tool.get("function", {}).get("name"))
        for tool in tools
        if tool.get("function", {}).get("name")
    ]
    events: list[dict[str, Any]] = []
    for event in record.get("tool_calls") or []:
        returned_names = list(event.get("returned_model_names") or [])
        result: dict[str, Any] = {}
        if returned_names:
            result = {
                "structuredContent": {
                    "summary": {
                        "candidates": [{"name": name} for name in returned_names],
                    }
                }
            }
        events.append(
            {
                "tool_call_id": event.get("tool_call_id"),
                "name": event.get("name"),
                "arguments": event.get("arguments") or {},
                "ok": bool(event.get("ok")),
                "result": result,
            }
        )
    successful = [event for event in events if event.get("ok")]
    question = record.get("question") or {}
    messages = record.get("messages") or []
    return {
        "id": record.get("id"),
        "status": "ok" if record.get("generation_status") == "ok" else "error",
        "format_version": FULL_TRAJECTORY_FORMAT_VERSION,
        "question_record": {
            "allowed_tools": tool_names,
            "task": {
                "model_catalog_scope": question.get("model_catalog_scope", "none"),
                "input_mode": question.get("input_mode", "text_only"),
            },
            "resources": {"dataset": {"path": ""}, "images": []},
        },
        "messages": messages,
        "tools": tools,
        "tool_events": events,
        "successful_tool_calls": len(successful),
        "distinct_successful_tools": sorted(
            {str(event.get("name")) for event in successful}
        ),
        "final_answer": _final_answer(messages),
    }


def deterministic_single_candidate_review(
    record: dict[str, Any],
    model_catalog: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if record.get("format_version") != TRAJECTORY_FORMAT_VERSION:
        return {
            "passed": False,
            "score": 0.0,
            "flags": ["unsupported_single_candidate_format"],
            "warnings": [],
            "observed": {
                "successful_calls": 0,
                "failed_calls": 0,
                "returned_model_names": [],
            },
        }
    if record.get("generation_status") != "ok":
        return {
            "passed": False,
            "score": 0.0,
            "flags": ["generation_error"],
            "warnings": [],
            "observed": {
                "successful_calls": 0,
                "failed_calls": 0,
                "returned_model_names": [],
            },
        }
    review = deterministic_trajectory_review(
        _review_record(record),
        model_catalog=model_catalog,
    )
    return {
        "passed": bool(review.get("passed")),
        "score": float(review.get("score", 0) or 0),
        "flags": list(review.get("flags") or []),
        "warnings": list(review.get("warnings") or []),
        "subscores": review.get("subscores") or {},
        "observed": review.get("observed") or {},
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
        if str(record["id"]) in completed:
            continue
        review = deterministic_single_candidate_review(record, model_catalog=model_catalog)
        output = {**record, "verification": review}
        append_jsonl(verified_path if review["passed"] else rejected_path, output)
