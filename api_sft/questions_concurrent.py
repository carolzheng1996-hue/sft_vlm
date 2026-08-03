"""Concurrent Question Writer runner.

This module is intentionally separate from :mod:`api_sft.questions`.  It
reuses the existing QuestionSpec contract, writer payload, validators and
runtime materializers, but schedules independent ``question_group`` calls in
bounded waves.  The original serial generator and its output files are never
modified by this runner.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from .api_client import OpenAICompatibleClient, parse_json_object
from .common import iter_jsonl, load_env_file, load_yaml, resolve_path, write_json, write_jsonl
from .questions import (
    QUESTION_AUDIT_FORMAT_VERSION,
    QUESTION_CONTRACT_VERSION,
    QUESTION_RUNTIME_FORMAT_VERSION,
    QUESTION_WRITER_PROMPT_ID,
    _coverage_rows,
    _expression_profile,
    _peer_questions,
    _task_specs,
    _validation_feedback,
    _with_diversity_quality,
    materialize_question_audit,
    materialize_question_record,
    question_writer_payload,
    validate_question,
    validate_question_diversity,
    write_coverage_report,
)


DEFAULT_OUTPUT_SUBDIR = "concurrent_questions"
OUTPUT_NAMES = (
    "questions.final.jsonl",
    "questions.audit.jsonl",
    "questions.rejected.jsonl",
    "coverage_report.json",
)


def _groups_in_input_order(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group specs while preserving the first-seen order from question_specs."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        group_id = str(row.get("question_group_id") or row["id"])
        if group_id not in grouped:
            grouped[group_id] = []
            order.append(group_id)
        grouped[group_id].append(row)
    return [grouped[group_id] for group_id in order]


def _paths(output_dir: Path) -> dict[str, Path]:
    return {
        "root": output_dir,
        "questions": output_dir / "questions.final.jsonl",
        "audit": output_dir / "questions.audit.jsonl",
        "rejected": output_dir / "questions.rejected.jsonl",
        "coverage": output_dir / "coverage_report.json",
        "archives": output_dir / "archives",
    }


def _archive_existing(paths: dict[str, Path], reason: str = "fresh_restart") -> Path | None:
    existing = [paths[key] for key in ("questions", "audit", "rejected", "coverage") if paths[key].exists()]
    if not existing:
        return None

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f%z")
    archive_dir = paths["archives"] / timestamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []
    moved: list[tuple[Path, Path]] = []
    try:
        for source in existing:
            target = archive_dir / source.name
            line_count = None
            if source.suffix == ".jsonl":
                line_count = sum(1 for _ in iter_jsonl(source))
            shutil.move(str(source), str(target))
            moved.append((source, target))
            entries.append({"source": str(source), "archived_as": source.name, "jsonl_records": line_count})
    except Exception:
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                shutil.move(str(target), str(source))
        raise

    write_json(
        archive_dir / "archive_manifest.json",
        {
            "archived_at": datetime.now().astimezone().isoformat(),
            "status": "stale_archived",
            "reason": reason,
            "entries": entries,
        },
    )
    return archive_dir


def _assert_resume_formats(paths: dict[str, Path]) -> None:
    final_path = paths["questions"]
    audit_path = paths["audit"]
    if not final_path.exists() and not audit_path.exists():
        return
    if not final_path.exists() or not audit_path.exists():
        raise RuntimeError("cannot resume concurrent questions without both final and audit files; use --fresh")

    final_versions = {str(row.get("format_version")) for row in iter_jsonl(final_path)}
    audits = list(iter_jsonl(audit_path))
    audit_versions = {str(row.get("format_version")) for row in audits}
    contracts = {str(row.get("question_contract_version")) for row in audits}
    writer_prompts = {str(row.get("question_writer_prompt_id")) for row in audits}
    if (
        final_versions - {QUESTION_RUNTIME_FORMAT_VERSION}
        or audit_versions - {QUESTION_AUDIT_FORMAT_VERSION}
        or contracts - {QUESTION_CONTRACT_VERSION}
        or writer_prompts - {QUESTION_WRITER_PROMPT_ID}
    ):
        raise RuntimeError("cannot resume stale question protocol; use --fresh")


def _candidate_from_response(
    raw: str,
    model_config: dict[str, Any],
    expression_profile: dict[str, str],
) -> dict[str, Any]:
    obj = parse_json_object(raw)
    return {
        "user_request": str(obj.get("user_request", "")).strip(),
        "decision_points": obj.get("decision_points", []),
        "constraint_key": str(obj.get("constraint_key", "")).strip(),
        "business_facts_used": obj.get("business_facts_used", []),
        "model": model_config.get("model"),
        "expression_profile_id": expression_profile["profile_id"],
        "expression_profile_version": expression_profile["version"],
    }


def _call_group(
    group: list[dict[str, Any]],
    model_config: dict[str, Any],
    writer_system: str,
    feedback: list[str],
    expression_profile: dict[str, str],
) -> dict[str, Any]:
    """Call the writer for one group.  No shared state or files are touched."""

    payload = question_writer_payload(group, feedback, expression_profile)
    client = OpenAICompatibleClient(model_config)
    try:
        raw, usage, latency = client.complete(
            [
                {"role": "system", "content": writer_system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
        generation = _candidate_from_response(raw, model_config, expression_profile)
        return {
            "status": "candidate",
            "generation": generation,
            "usage": usage,
            "latency_seconds": round(latency, 3),
        }
    except Exception as exc:  # noqa: BLE001 - preserve per-group failure for retry
        return {
            "status": "error",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "usage": {},
            "latency_seconds": None,
        }


def _quality_for_generation(
    generation: dict[str, Any],
    group: list[dict[str, Any]],
    existing: dict[str, dict[str, Any]],
    spec_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    first = group[0]
    quality = validate_question(generation["user_request"], first, generation)
    group_id = str(first.get("question_group_id") or first["id"])
    diversity = validate_question_diversity(
        generation["user_request"],
        _peer_questions(existing, spec_by_id, group_id),
    )
    generation["diversity"] = diversity
    return _with_diversity_quality(quality, diversity), generation


def _reusable_generation(
    reusable: dict[str, Any],
    audit: dict[str, Any],
    expression_profile: dict[str, str],
) -> dict[str, Any]:
    metadata = audit.get("question_generation", {})
    return {
        "user_request": reusable.get("prompt", {}).get("user_request", ""),
        "decision_points": metadata.get("decision_points", []),
        "constraint_key": metadata.get("constraint_key", ""),
        "business_facts_used": metadata.get("business_facts_used", []),
        "model": metadata.get("model"),
        "expression_profile_id": expression_profile["profile_id"],
        "expression_profile_version": expression_profile["version"],
    }


def _materialize_missing(
    group: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    generation: dict[str, Any],
    quality: dict[str, Any],
    attempts: list[dict[str, Any]],
    reusable_id: str | None,
    existing: dict[str, dict[str, Any]],
    accepted_audits: dict[str, dict[str, Any]],
) -> None:
    for index, spec in enumerate(missing):
        out = materialize_question_record(spec, generation)
        audit = materialize_question_audit(
            spec,
            generation,
            quality,
            attempts if index == 0 else [],
            reusable_id if reusable_id else (None if index == 0 else missing[0]["id"]),
        )
        existing[out["id"]] = out
        accepted_audits[out["id"]] = audit


def _write_checkpoint(
    paths: dict[str, Path],
    existing: dict[str, dict[str, Any]],
    accepted_audits: dict[str, dict[str, Any]],
    rejected: dict[str, dict[str, Any]],
    specs: dict[str, dict[str, Any]],
) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    rows = [existing[key] for key in sorted(existing)]
    audits = [accepted_audits[key] for key in sorted(accepted_audits)]
    write_jsonl(paths["questions"], rows)
    write_jsonl(paths["audit"], audits)
    if rejected:
        write_jsonl(paths["rejected"], [rejected[key] for key in sorted(rejected)])
    elif paths["rejected"].exists():
        paths["rejected"].unlink()

    task_specs = _task_specs(Path(__file__).with_name("task_pool.yaml"))
    all_tools = {name for spec in task_specs for name in spec["preferred_tools"]}
    coverage_rows = _coverage_rows(specs, rows, accepted_audits)
    write_coverage_report(
        coverage_rows,
        paths["coverage"],
        task_specs,
        all_tools,
        False,
    )


def generate_questions_concurrent(
    specs_path: Path,
    output_dir: Path,
    model_config: dict[str, Any],
    writer_prompt_path: Path,
    max_concurrency: int = 4,
    resume: bool = False,
    fresh: bool = False,
    limit: int | None = None,
    max_attempts: int = 3,
) -> list[dict[str, Any]]:
    """Generate Questions concurrently, preserving the serial output contract."""

    if resume and fresh:
        raise ValueError("--resume and --fresh are mutually exclusive")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    paths = _paths(output_dir)
    output_exists = any(paths[key].exists() for key in ("questions", "audit", "rejected", "coverage"))
    if fresh:
        _archive_existing(paths)
    elif resume:
        _assert_resume_formats(paths)
    elif output_exists:
        raise RuntimeError("concurrent question outputs already exist; use --resume or --fresh")

    output_dir.mkdir(parents=True, exist_ok=True)
    specs_list = list(iter_jsonl(specs_path))
    spec_by_id = {row["id"]: row for row in specs_list}
    all_existing = (
        {row["id"]: row for row in iter_jsonl(paths["questions"])}
        if paths["questions"].exists()
        else {}
    )
    all_audits = (
        {row["id"]: row for row in iter_jsonl(paths["audit"])}
        if paths["audit"].exists()
        else {}
    )

    existing: dict[str, dict[str, Any]] = {}
    accepted_audits: dict[str, dict[str, Any]] = {}
    invalid_existing_ids: set[str] = set()
    for row_id in sorted(all_existing):
        row = all_existing[row_id]
        spec = spec_by_id.get(row_id)
        audit = all_audits.get(row_id, {})
        metadata = audit.get("question_generation", {})
        user_request = str(row.get("prompt", {}).get("user_request", ""))
        quality = validate_question(user_request, spec, metadata) if spec else {"passed": False}
        if spec:
            group_id = str(spec.get("question_group_id") or row_id)
            diversity = validate_question_diversity(
                user_request,
                _peer_questions(existing, spec_by_id, group_id),
            )
            quality = _with_diversity_quality(quality, diversity)
        reusable = bool(
            spec
            and set(row) == {"id", "format_version", "spec_ref", "task", "prompt", "resources", "allowed_tools"}
            and row.get("format_version") == QUESTION_RUNTIME_FORMAT_VERSION
            and row.get("prompt", {}).get("system_prompt")
            and row.get("spec_ref", {}).get("version") == spec.get("question_spec_version")
            and row.get("spec_ref", {}).get("hash") == spec.get("question_spec_hash")
            and audit.get("format_version") == QUESTION_AUDIT_FORMAT_VERSION
            and audit.get("question_contract_version") == QUESTION_CONTRACT_VERSION
            and audit.get("question_writer_prompt_id") == QUESTION_WRITER_PROMPT_ID
            and audit.get("spec_ref", {}).get("hash") == spec.get("question_spec_hash")
            and quality["passed"]
        )
        if reusable:
            existing[row_id] = row
            refreshed_audit = dict(audit)
            refreshed_audit["question_quality"] = quality
            accepted_audits[row_id] = refreshed_audit
        else:
            invalid_existing_ids.add(row_id)

    # Preserve the deterministic order already materialized in question_specs.
    # Sorting by group_id would put every ``pair_*`` group before every
    # ``scenario_*`` standalone group and make small --limit runs look
    # incorrectly 100% paired.
    groups = _groups_in_input_order(specs_list)
    if limit is not None:
        selected = groups[:limit]
        selected_keys = {str(group[0].get("question_group_id") or group[0]["id"]) for group in selected}
        repair_groups = [
            group
            for group in groups[limit:]
            if any(spec["id"] in invalid_existing_ids for spec in group)
            and str(group[0].get("question_group_id") or group[0]["id"]) not in selected_keys
        ]
        groups = selected + repair_groups

    writer_system = writer_prompt_path.read_text(encoding="utf-8").strip()
    rejected: dict[str, dict[str, Any]] = {}
    if paths["rejected"].exists() and resume:
        rejected = {row["id"]: row for row in iter_jsonl(paths["rejected"])}

    # Complete partially materialized groups without spending another API call.
    pending: deque[dict[str, Any]] = deque()
    for group in groups:
        missing = [spec for spec in group if spec["id"] not in existing]
        if not missing:
            continue
        group_id = str(group[0].get("question_group_id") or group[0]["id"])
        expression_profile = _expression_profile(group)
        reusable_spec = next((spec for spec in group if spec["id"] in existing), None)
        if reusable_spec is not None:
            reusable_audit = accepted_audits[reusable_spec["id"]]
            generation = _reusable_generation(existing[reusable_spec["id"]], reusable_audit, expression_profile)
            quality, generation = _quality_for_generation(generation, group, existing, spec_by_id)
            if quality["passed"]:
                _materialize_missing(
                    group,
                    missing,
                    generation,
                    quality,
                    [],
                    reusable_spec["id"],
                    existing,
                    accepted_audits,
                )
                for spec in missing:
                    rejected.pop(spec["id"], None)
            else:
                for spec in missing:
                    rejected[spec["id"]] = {
                        "id": spec["id"],
                        "stage": "question_generation",
                        "reason": "reusable_group_failed_validation",
                        "attempts": [],
                        "question_spec": spec,
                    }
            continue
        pending.append(
            {
                "group": group,
                "missing": missing,
                "attempt": 0,
                "feedback": [],
                "attempts": [],
                "group_id": group_id,
            }
        )

    _write_checkpoint(paths, existing, accepted_audits, rejected, spec_by_id)

    while pending:
        batch: list[dict[str, Any]] = [pending.popleft() for _ in range(min(max_concurrency, len(pending)))]
        futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            for state in batch:
                state["attempt"] += 1
                profile = _expression_profile(state["group"])
                future = pool.submit(
                    _call_group,
                    state["group"],
                    model_config,
                    writer_system,
                    state["feedback"],
                    profile,
                )
                futures[future] = state

            results: dict[str, dict[str, Any]] = {}
            for future in as_completed(futures):
                state = futures[future]
                try:
                    results[state["group_id"]] = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve worker isolation
                    results[state["group_id"]] = {
                        "status": "error",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "usage": {},
                        "latency_seconds": None,
                    }

        retry_states: list[dict[str, Any]] = []
        for state in sorted(batch, key=lambda item: item["group_id"]):
            result = results[state["group_id"]]
            attempts = state["attempts"]
            if result["status"] == "candidate":
                generation = result["generation"]
                quality, generation = _quality_for_generation(generation, state["group"], existing, spec_by_id)
                attempt_record = {
                    "attempt": state["attempt"],
                    "status": "accepted" if quality["passed"] else "validation_failed",
                    "usage": result.get("usage", {}),
                    "latency_seconds": result.get("latency_seconds"),
                    "quality": quality,
                }
                attempts.append(attempt_record)
                if quality["passed"]:
                    _materialize_missing(
                        state["group"],
                        state["missing"],
                        generation,
                        quality,
                        attempts,
                        None,
                        existing,
                        accepted_audits,
                    )
                    for spec in state["missing"]:
                        rejected.pop(spec["id"], None)
                    continue
                state["feedback"] = _validation_feedback(quality)
            else:
                attempts.append(
                    {
                        "attempt": state["attempt"],
                        "status": "error",
                        "error": result.get("error", "unknown worker error"),
                        "usage": result.get("usage", {}),
                        "latency_seconds": result.get("latency_seconds"),
                    }
                )
                state["feedback"] = [
                    f"上一次输出无法解析或请求失败：{result.get('error_type', 'RuntimeError')}。请严格输出所需 JSON 字段。"
                ]

            if state["attempt"] < max_attempts:
                retry_states.append(state)
            else:
                for spec in state["missing"]:
                    rejected[spec["id"]] = {
                        "id": spec["id"],
                        "stage": "question_generation",
                        "reason": "writer_failed_after_retries",
                        "attempts": attempts,
                        "question_spec": spec,
                    }

        # Retry failures before taking new groups, while keeping deterministic order.
        pending.extendleft(reversed(sorted(retry_states, key=lambda item: item["group_id"])))
        _write_checkpoint(paths, existing, accepted_audits, rejected, spec_by_id)

    _write_checkpoint(paths, existing, accepted_audits, rejected, spec_by_id)
    return [existing[key] for key in sorted(existing)]


def _load_cli_config(config_path: Path) -> tuple[dict[str, Any], Path, Path, dict[str, Any]]:
    config = load_yaml(config_path.resolve())
    workspace = Path.cwd().resolve()
    env_file = config.get("env_file")
    if env_file:
        load_env_file(resolve_path(env_file, workspace), bool(config.get("env_override", False)))
    root = resolve_path(config.get("output_dir", "api_sft/output"), workspace)
    generation = config.get("generation", {})
    return config, workspace, root, generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Question records with bounded concurrent Question Writer calls")
    parser.add_argument("--config", type=Path, default=Path("api_sft/config.yaml"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.resume and args.fresh:
        parser.error("--resume and --fresh are mutually exclusive")

    config, workspace, root, generation = _load_cli_config(args.config)
    output_dir = args.output_dir.resolve() if args.output_dir else root / DEFAULT_OUTPUT_SUBDIR
    specs_path = root / "question_specs.jsonl"
    writer_path = resolve_path(
        generation.get("question_writer_prompt", "api_sft/prompts/question_writer_system.txt"),
        workspace,
    )
    model_config = dict(config["models"]["question"])
    model_config.setdefault("timeout_seconds", int(generation.get("timeout_seconds", 120)))
    model_config.setdefault("retries", int(generation.get("retries", 3)))
    max_concurrency = (
        args.max_concurrency
        if args.max_concurrency is not None
        else int(generation.get("max_concurrency", 4))
    )
    rows = generate_questions_concurrent(
        specs_path,
        output_dir,
        model_config,
        writer_path,
        max_concurrency=max_concurrency,
        resume=args.resume,
        fresh=args.fresh,
        limit=args.limit,
        max_attempts=int(generation.get("question_max_attempts", 3)),
    )
    print(
        json.dumps(
            {
                "questions": len(rows),
                "output": str(output_dir / "questions.final.jsonl"),
                "audit": str(output_dir / "questions.audit.jsonl"),
                "rejected": str(output_dir / "questions.rejected.jsonl"),
                "coverage": str(output_dir / "coverage_report.json"),
                "max_concurrency": max_concurrency,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
