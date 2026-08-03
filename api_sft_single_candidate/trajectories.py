from __future__ import annotations

import json
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from api_sft.api_client import APIRequestError, OpenAICompatibleClient
from api_sft.common import iter_jsonl, stable_hash
from api_sft.compact_json import append_jsonl, json_safe, write_jsonl
from api_sft.questions import QUESTION_RUNTIME_FORMAT_VERSION
from api_sft.tool_runtime import ClaudeTsaToolRuntime
from api_sft.trajectories import (
    TrajectoryGenerationError,
    _model_catalog_search_definition,
    _runtime_scope,
    generate_one_trajectory,
)
from api_sft.trajectories_compact import (
    _candidate_audit_view,
    _final_answer,
    _metrics,
    _normalize_messages,
    _question_view,
    _review_view,
    _tool_call_views,
)
from api_sft.trajectory_verify import deterministic_trajectory_review

from .provider_adapter import (
    ProviderCompatibleClient,
    SessionBinding,
    SessionBoundRuntime,
    ToolSchemaCompatibilityError,
    model_tools_from_execution,
    project_tools_for_api,
    sanitize_candidate_record,
)


TRAJECTORY_FORMAT_VERSION = "tool_trajectory_single_v2_compact"
GENERATION_AUDIT_FORMAT_VERSION = "trajectory_generation_audit_single_v2_compact"
GLOBAL_GENERATION_ERRORS = (APIRequestError, ToolSchemaCompatibilityError)


def _usage_add(total: dict[str, int], usage: dict[str, Any]) -> None:
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        total[key] = total.get(key, 0) + int(usage.get(key, 0) or 0)


def _global_generation_error(exc: BaseException) -> Exception | None:
    """Recover a global boundary failure wrapped by trajectory snapshot handling."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, GLOBAL_GENERATION_ERRORS):
            return current
        current = current.__cause__ or current.__context__
    return None


def validate_input_tool_schemas(
    input_path: Path,
    repo_root: Path,
    limit: int | None = None,
) -> int:
    """Validate every tool needed by the selected questions without an API request."""

    rows = list(iter_jsonl(input_path))
    rows = rows[:limit] if limit else rows
    allowed_names = {
        str(name)
        for row in rows
        for name in row.get("allowed_tools", [])
        if str(name)
    }
    with tempfile.TemporaryDirectory(prefix="single_candidate_schema_") as temp_dir:
        runtime = ClaudeTsaToolRuntime(repo_root, Path(temp_dir))
        described = runtime.describe_tools(allowed_names, session_id="schema_validation")
    observed_names = {str(item.get("name")) for item in described}
    missing = sorted(allowed_names - observed_names)
    if missing:
        raise ToolSchemaCompatibilityError(
            "Tools missing from live claude_tsa registry: " + ", ".join(missing)
        )
    execution_tools = [
        {
            "type": "function",
            "function": {
                "name": item["name"],
                "description": item["description"],
                "parameters": item["input_schema"],
            },
        }
        for item in described
    ]
    scopes = {_runtime_scope(row) for row in rows} - {"none"}
    for scope in sorted(scopes):
        if scope not in {"forecast", "anomaly_detection"}:
            raise ToolSchemaCompatibilityError(f"Unsupported model_catalog_scope: {scope}")
        execution_tools.append(_model_catalog_search_definition("schema_validation", scope))
    model_tools = model_tools_from_execution(execution_tools)
    project_tools_for_api(model_tools)
    return len(model_tools)


def _generate_candidate(
    row: dict[str, Any],
    model_config: dict[str, Any],
    repo_root: Path,
    workspace_root: Path,
    model_catalog: list[dict[str, Any]],
    max_attempts: int,
    min_tool_calls: int,
    min_distinct_tools: int,
    max_tool_calls: int,
    max_tool_result_chars: int,
) -> dict[str, Any]:
    """Generate exactly one candidate, retrying only terminal generation failures."""

    base_client = OpenAICompatibleClient(model_config)
    attempt_errors: list[dict[str, Any]] = []
    failed_attempts: list[dict[str, Any]] = []
    try:
        for attempt in range(1, max_attempts + 1):
            binding = SessionBinding()
            adapter = ProviderCompatibleClient(base_client, binding)
            try:
                question_key = stable_hash(str(row["id"]))[:16]
                attempt_workspace = (
                    workspace_root
                    / f"question_{question_key}"
                    / f"attempt_{attempt:02d}"
                )
                base_runtime = ClaudeTsaToolRuntime(repo_root, attempt_workspace)
                runtime = SessionBoundRuntime(base_runtime, binding)
                record = generate_one_trajectory(
                    row,
                    adapter,
                    runtime,
                    model_catalog=model_catalog,
                    min_tool_calls=min_tool_calls,
                    min_distinct_tools=min_distinct_tools,
                    max_tool_calls=max_tool_calls,
                    max_tool_result_chars=max_tool_result_chars,
                    candidate_index=0,
                )
                record = sanitize_candidate_record(record, binding, adapter.audit_summary())
                record["generation_attempt"] = attempt
                record["prior_attempt_errors"] = attempt_errors
                record["failed_attempts"] = failed_attempts
                return record
            except TrajectoryGenerationError as exc:
                global_error = _global_generation_error(exc)
                if global_error is not None:
                    raise global_error
                partial = sanitize_candidate_record(
                    dict(exc.partial_record),
                    binding,
                    adapter.audit_summary(),
                )
                partial["attempt"] = attempt
                partial["error_type"] = exc.error_type
                partial["error"] = str(exc)
                failed_attempts.append(partial)
                attempt_errors.append(
                    {"attempt": attempt, "error_type": exc.error_type, "error": str(exc)}
                )
            except Exception as exc:  # noqa: BLE001 - every failed attempt is audited
                global_error = _global_generation_error(exc)
                if global_error is not None:
                    raise global_error
                failed_attempts.append(
                    {
                        "attempt": attempt,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "messages": [],
                        "tool_events": [],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                        "provider_adapter": adapter.audit_summary(),
                    }
                )
                attempt_errors.append(
                    {"attempt": attempt, "error_type": type(exc).__name__, "error": str(exc)}
                )
    finally:
        close = getattr(base_client, "close", None)
        if callable(close):
            close()

    last_partial = dict(failed_attempts[-1]) if failed_attempts else {}
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for failed in failed_attempts:
        _usage_add(total_usage, failed.get("usage") or {})
    return {
        **last_partial,
        "id": row["id"],
        "status": "error",
        "question_record": row,
        "generation_attempt": max_attempts,
        "attempt_errors": attempt_errors,
        "failed_attempts": failed_attempts,
        "model": model_config.get("model"),
        "candidate_index": 0,
        "usage": total_usage,
        "tool_events": list(last_partial.get("tool_events") or []),
    }


def _worker_error_candidate(
    row: dict[str, Any],
    model_config: dict[str, Any],
    max_attempts: int,
    exc: Exception,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a reviewable error result when a worker fails outside generation retry handling."""

    error_type = type(exc).__name__
    candidate = {
        "id": row["id"],
        "status": "error",
        "format_version": TRAJECTORY_FORMAT_VERSION,
        "question_record": row,
        "model": model_config.get("model"),
        "candidate_index": 0,
        "generation_attempt": max_attempts,
        "error_type": error_type,
        "error": str(exc),
        "failed_attempts": [],
        "tool_events": [],
        "messages": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    review = {
        "passed": False,
        "score": 0,
        "flags": ["worker_exception"],
        "warnings": [],
        "observed": {"error_type": error_type},
    }
    return candidate, review


def _process_row(
    row: dict[str, Any],
    candidate_config: dict[str, Any],
    repo_root: Path,
    workspace_root: Path,
    model_catalog: list[dict[str, Any]],
    max_attempts: int,
    min_tool_calls: int,
    min_distinct_tools: int,
    max_tool_calls: int,
    max_tool_result_chars: int,
) -> dict[str, Any]:
    """Generate, review, and materialize one question without touching shared files."""

    try:
        candidate = _generate_candidate(
            row,
            dict(candidate_config),
            repo_root,
            workspace_root,
            model_catalog,
            max_attempts,
            min_tool_calls,
            min_distinct_tools,
            max_tool_calls,
            max_tool_result_chars,
        )
        review = deterministic_trajectory_review(candidate, model_catalog=model_catalog)
    except GLOBAL_GENERATION_ERRORS:
        raise
    except Exception as exc:  # noqa: BLE001 - isolate one failed question from the pool
        candidate, review = _worker_error_candidate(row, candidate_config, max_attempts, exc)

    audit = _audit_record(row, candidate, review)
    output = (
        _accepted_record(row, candidate, review)
        if review.get("passed")
        else _rejected_record(row, candidate, review)
    )
    return {
        "id": str(row["id"]),
        "audit": audit,
        "output": output,
        "accepted": bool(review.get("passed")),
    }


def _read_unique_records(path: Path) -> dict[str, dict[str, Any]]:
    """Read a JSONL checkpoint, keeping the last record for each ID."""

    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if "id" not in row:
            continue
        records[str(row["id"])] = row
    return records


def _reconcile_resume_files(output_path: Path, audit_path: Path) -> set[str]:
    """Repair one-sided/duplicate checkpoints before resuming generation."""

    outputs = _read_unique_records(output_path)
    audits = _read_unique_records(audit_path)
    completed = set(outputs) & set(audits)
    if output_path.exists():
        write_jsonl(output_path, (outputs[row_id] for row_id in sorted(completed)))
    if audit_path.exists():
        write_jsonl(audit_path, (audits[row_id] for row_id in sorted(completed)))
    return completed


def _assert_resume_model(output_path: Path, model: str) -> None:
    observed = {
        str(row.get("model"))
        for row in _read_unique_records(output_path).values()
        if row.get("model")
    }
    mismatched = sorted(observed - {str(model)})
    if mismatched:
        raise RuntimeError(
            "Cannot resume trajectories generated by a different model: "
            + ", ".join(mismatched)
            + "; use --fresh"
        )


def _generation_view(candidate: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "single_candidate",
        "candidate_index": 0,
        "candidate_model": candidate.get("model"),
        "attempts": int(candidate.get("generation_attempt", 1) or 1),
        "hard_review": _review_view(review),
        "audit_id": candidate.get("id"),
    }


def _accepted_record(
    row: dict[str, Any],
    candidate: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    return json_safe(
        {
            "id": row["id"],
            "format_version": TRAJECTORY_FORMAT_VERSION,
            "question": _question_view(row, candidate),
            "generation_status": "ok",
            "model": candidate.get("model"),
            "messages": _normalize_messages(candidate.get("messages") or []),
            "tools": json_safe(candidate.get("tools") or []),
            "tool_calls": _tool_call_views(candidate),
            "final_answer": _final_answer(candidate),
            "metrics": _metrics(candidate),
            "generation": _generation_view(candidate, review),
        }
    )


def _rejected_record(
    row: dict[str, Any],
    candidate: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    generated = candidate.get("status") == "ok"
    flags = list(review.get("flags") or [])
    return json_safe(
        {
            "id": row["id"],
            "format_version": TRAJECTORY_FORMAT_VERSION,
            "question": _question_view(row, candidate),
            "generation_status": "error",
            "model": candidate.get("model"),
            "error": {
                "type": "hard_review_failed" if generated else str(
                    candidate.get("error_type")
                    or candidate.get("terminal_error_type")
                    or "generation_error"
                ),
                "message": (
                    "Single candidate failed deterministic review: " + ", ".join(flags)
                    if generated
                    else str(
                        candidate.get("error")
                        or candidate.get("terminal_error")
                        or "Single candidate generation failed."
                    )
                ),
            },
            "generation": _generation_view(candidate, review),
        }
    )


def _audit_record(
    row: dict[str, Any],
    candidate: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    candidate_view = _candidate_audit_view(candidate, review)
    if candidate.get("provider_adapter"):
        candidate_view["provider_adapter"] = candidate["provider_adapter"]
    for view, attempt in zip(
        candidate_view.get("failed_attempts") or [],
        candidate.get("failed_attempts") or [],
    ):
        if attempt.get("provider_adapter"):
            view["provider_adapter"] = attempt["provider_adapter"]
    return json_safe(
        {
            "id": row["id"],
            "format_version": GENERATION_AUDIT_FORMAT_VERSION,
            "question": _question_view(row, candidate),
            "candidate": candidate_view,
            "decision": {
                "rule": "single_candidate_hard_gate",
                "accepted": bool(review.get("passed")),
            },
        }
    )


def generate_trajectories(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
    candidate_config: dict[str, Any],
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
    max_concurrency: int = 4,
) -> None:
    """Generate and hard-review one real tool trajectory per question concurrently."""

    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if not resume:
        for path in [output_path, audit_path]:
            if path.exists():
                path.unlink()
    rows = list(iter_jsonl(input_path))
    rows = rows[:limit] if limit else rows
    row_ids = [str(row["id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("Question input contains duplicate IDs")
    stale = [row.get("id") for row in rows if row.get("format_version") != QUESTION_RUNTIME_FORMAT_VERSION]
    if stale:
        raise RuntimeError(
            f"Single-candidate trajectories require {QUESTION_RUNTIME_FORMAT_VERSION}: "
            + ", ".join(map(str, stale[:3]))
        )

    if resume:
        _assert_resume_model(output_path, str(candidate_config.get("model") or ""))
    all_completed = _reconcile_resume_files(output_path, audit_path) if resume else set()
    selected_ids = set(row_ids)
    completed = all_completed & selected_ids
    pending = [row for row in rows if str(row["id"]) not in completed]
    total = len(rows)
    already_completed = total - len(pending)
    completed_count = already_completed
    accepted_count = sum(
        1
        for row in _read_unique_records(output_path).values()
        if str(row.get("id")) in completed and row.get("generation_status") == "ok"
    )
    rejected_count = already_completed - accepted_count
    started = time.monotonic()
    in_flight: dict[Any, str] = {}

    def print_progress(last_id: str | None = None) -> None:
        percent = round((completed_count / total) * 100, 1) if total else 100.0
        payload = {
            "status": "progress",
            "completed": completed_count,
            "total": total,
            "percent": percent,
            "accepted": accepted_count,
            "rejected": rejected_count,
            "in_flight": len(in_flight),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        if last_id is not None:
            payload["last_id"] = last_id
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    if not pending:
        print_progress()
        return

    def write_result(result: dict[str, Any]) -> None:
        nonlocal completed_count, accepted_count, rejected_count
        append_jsonl(audit_path, result["audit"])
        append_jsonl(output_path, result["output"])
        completed_count += 1
        if result["accepted"]:
            accepted_count += 1
        else:
            rejected_count += 1
        print_progress(result["id"])

    canary = pending[0]
    try:
        canary_result = _process_row(
            canary,
            dict(candidate_config),
            repo_root,
            workspace_root,
            model_catalog,
            max_attempts,
            min_tool_calls,
            min_distinct_tools,
            max_tool_calls,
            max_tool_result_chars,
        )
    except GLOBAL_GENERATION_ERRORS as exc:
        payload: dict[str, Any] = {
            "status": "aborted",
            "stage": "canary",
            "completed": completed_count,
            "total": total,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if isinstance(exc, APIRequestError):
            payload["api_error"] = exc.audit_summary()
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        raise RuntimeError(
            f"Global API compatibility check failed on canary {canary['id']}: {exc}"
        ) from exc
    write_result(canary_result)
    remaining = pending[1:]
    if not remaining:
        return

    def submit_next(pool: ThreadPoolExecutor, next_index: int, in_flight: dict[Any, str]) -> int:
        if next_index >= len(remaining):
            return next_index
        row = remaining[next_index]
        future = pool.submit(
            _process_row,
            row,
            dict(candidate_config),
            repo_root,
            workspace_root,
            model_catalog,
            max_attempts,
            min_tool_calls,
            min_distinct_tools,
            max_tool_calls,
            max_tool_result_chars,
        )
        in_flight[future] = str(row["id"])
        return next_index + 1

    pool = ThreadPoolExecutor(max_workers=max_concurrency)
    next_index = 0
    fatal_error: Exception | None = None
    try:
        while next_index < len(remaining) and len(in_flight) < max_concurrency:
            next_index = submit_next(pool, next_index, in_flight)

        while in_flight:
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                row_id = in_flight.pop(future)
                try:
                    result = future.result()
                except GLOBAL_GENERATION_ERRORS as exc:
                    fatal_error = fatal_error or exc
                    continue
                except Exception as exc:  # noqa: BLE001 - final worker isolation guard
                    row = next(row for row in remaining if str(row["id"]) == row_id)
                    candidate, review = _worker_error_candidate(
                        row, candidate_config, max_attempts, exc
                    )
                    result = {
                        "id": row_id,
                        "audit": _audit_record(row, candidate, review),
                        "output": _rejected_record(row, candidate, review),
                        "accepted": False,
                    }
                write_result(result)
            while (
                fatal_error is None
                and next_index < len(remaining)
                and len(in_flight) < max_concurrency
            ):
                next_index = submit_next(pool, next_index, in_flight)
    finally:
        pool.shutdown(wait=True)
    if fatal_error is not None:
        payload = {
            "status": "aborted",
            "stage": "concurrent_generation",
            "completed": completed_count,
            "total": total,
            "error_type": type(fatal_error).__name__,
            "error": str(fatal_error),
        }
        if isinstance(fatal_error, APIRequestError):
            payload["api_error"] = fatal_error.audit_summary()
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        raise RuntimeError(
            f"Global API failure stopped further question submission: {fatal_error}"
        ) from fatal_error
