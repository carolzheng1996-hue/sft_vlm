from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from api_sft.common import iter_jsonl, load_yaml, read_json, resolve_path
from api_sft.compact_json import write_json

from .exporters import export_trajectory_datasets
from .trajectories import (
    GENERATION_AUDIT_FORMAT_VERSION,
    TRAJECTORY_FORMAT_VERSION,
    generate_trajectories,
    validate_input_tool_schemas,
)
from .verify import verify_trajectories


def paths(config: dict[str, Any], workspace: Path) -> dict[str, Path]:
    source = config.get("source") or {}
    root = resolve_path(config.get("output_dir", "api_sft/output_single_candidate"), workspace)
    return {
        "questions": resolve_path(
            source.get(
                "questions",
                "api_sft/output/concurrent_questions/questions.final.jsonl",
            ),
            workspace,
        ),
        "model_catalog": resolve_path(
            source.get("model_catalog", "api_sft/output/catalogs/models.normalized.json"),
            workspace,
        ),
        "root": root,
        "workspaces": root / "trajectory_workspaces",
        "raw": root / "trajectories.raw.jsonl",
        "audit": root / "trajectories.generation_audit.jsonl",
        "verified": root / "trajectories.verified.jsonl",
        "rejected": root / "trajectories.rejected.jsonl",
        "exports": root / "trajectory_exports",
        "archives": root / "trajectory_run_archives",
    }


def load(config_path: Path) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    config = load_yaml(config_path.resolve())
    workspace = Path.cwd().resolve()
    return config, workspace, paths(config, workspace)


def _output_paths(p: dict[str, Path]) -> list[Path]:
    return [p["raw"], p["audit"], p["verified"], p["rejected"], p["workspaces"], p["exports"]]


def archive_run(p: dict[str, Path], reason: str = "fresh_restart") -> Path | None:
    existing = [path for path in _output_paths(p) if path.exists()]
    if not existing:
        return None
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f%z")
    archive_dir = p["archives"] / timestamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []
    moved: list[tuple[Path, Path]] = []
    try:
        for source in existing:
            if source.is_file():
                records = sum(1 for _ in iter_jsonl(source)) if source.suffix == ".jsonl" else None
                files = 1
            else:
                records = None
                files = sum(1 for item in source.rglob("*") if item.is_file())
            target = archive_dir / source.name
            shutil.move(str(source), str(target))
            moved.append((source, target))
            entries.append(
                {
                    "source": str(source),
                    "archived_as": source.name,
                    "file_count": files,
                    "jsonl_records": records,
                }
            )
    except Exception:
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                shutil.move(str(target), str(source))
        if archive_dir.exists() and not any(archive_dir.iterdir()):
            archive_dir.rmdir()
        raise
    write_json(
        archive_dir / "archive_manifest.json",
        {
            "archived_at": datetime.now().astimezone().isoformat(),
            "status": "stale_archived",
            "reason": reason,
            "trajectory_format_version": TRAJECTORY_FORMAT_VERSION,
            "generation_audit_format_version": GENERATION_AUDIT_FORMAT_VERSION,
            "entries": entries,
        },
    )
    return archive_dir


def _assert_resumable_formats(p: dict[str, Path]) -> None:
    expected = {
        p["raw"]: TRAJECTORY_FORMAT_VERSION,
        p["audit"]: GENERATION_AUDIT_FORMAT_VERSION,
        p["verified"]: TRAJECTORY_FORMAT_VERSION,
        p["rejected"]: TRAJECTORY_FORMAT_VERSION,
    }
    for path, version in expected.items():
        if not path.exists():
            continue
        observed = {str(row.get("format_version")) for row in iter_jsonl(path)}
        if observed - {version}:
            raise RuntimeError(
                f"Cannot resume stale single-candidate format in {path}: "
                f"expected {version}, observed {sorted(observed)}; use --fresh"
            )


def _candidate_config(config: dict[str, Any]) -> dict[str, Any]:
    candidate = (config.get("models") or {}).get("candidate")
    if not isinstance(candidate, dict):
        raise RuntimeError("models.candidate must be one tool-call-capable model object")
    base_url = str(candidate.get("base_url") or "").strip()
    model = str(candidate.get("model") or "").strip()
    api_key = candidate.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeError(
            "models.candidate.base_url, models.candidate.model, and "
            "models.candidate.api_key are required"
        )
    if not base_url or not model:
        raise RuntimeError(
            "models.candidate.base_url, models.candidate.model, and "
            "models.candidate.api_key are required"
        )
    if ".example" in base_url or "replace-with" in model:
        raise RuntimeError("models.candidate still contains a placeholder endpoint/model")
    normalized_key = api_key.strip().lower()
    if normalized_key.startswith("replace-with") or normalized_key in {
        "your-api-key",
        "api-key",
        "changeme",
    }:
        raise RuntimeError("models.candidate.api_key still contains a placeholder value")
    if "provider" in candidate:
        raise RuntimeError(
            "models.candidate.provider is no longer used; remove it because all "
            "candidate models use the OpenAI-compatible API"
        )
    if "api_key_env" in candidate:
        raise RuntimeError(
            "models.candidate.api_key_env is no longer supported; set api_key directly"
        )
    trajectory = config.get("trajectory_generation") or {}
    result = dict(candidate)
    result["api_key"] = api_key.strip()
    result.setdefault("timeout_seconds", int(trajectory.get("timeout_seconds", 180)))
    result.setdefault("retries", int(trajectory.get("retries", 3)))
    result["response_format"] = False
    result["_omit_default_temperature"] = True
    result["_reuse_http_client"] = True
    result["_tools_preprojected"] = True
    return result


def _repo_root(config: dict[str, Any], workspace: Path) -> Path:
    value = (config.get("source") or {}).get("tools_repo_root")
    if not value:
        raise RuntimeError("source.tools_repo_root is required for live tool trajectories")
    return resolve_path(value, workspace)


def _model_catalog(p: dict[str, Path]) -> list[dict[str, Any]]:
    value = read_json(p["model_catalog"])
    models = value.get("models") if isinstance(value, dict) else None
    if not isinstance(models, list):
        raise RuntimeError(f"Invalid normalized model catalog: {p['model_catalog']}")
    return models


def do_generate(
    config: dict[str, Any],
    workspace: Path,
    p: dict[str, Path],
    resume: bool,
    fresh: bool,
    limit: int | None,
    max_concurrency: int | None,
) -> None:
    if resume and fresh:
        raise RuntimeError("--resume and --fresh are mutually exclusive")
    trajectory = config.get("trajectory_generation") or {}
    configured_concurrency = int(trajectory.get("max_concurrency", 4))
    concurrency = configured_concurrency if max_concurrency is None else int(max_concurrency)
    if concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    existing = [path for path in _output_paths(p) if path.exists()]
    if resume:
        _assert_resumable_formats(p)
    elif not fresh and existing:
        raise RuntimeError(
            "single-candidate outputs already exist; use --resume or --fresh"
        )
    if not p["questions"].is_file():
        raise FileNotFoundError(f"Question input not found: {p['questions']}")
    if not p["model_catalog"].is_file():
        raise FileNotFoundError(f"Model catalog not found: {p['model_catalog']}")
    candidate_config = _candidate_config(config)
    model_catalog = _model_catalog(p)
    repo_root = _repo_root(config, workspace)
    validated_tools = validate_input_tool_schemas(p["questions"], repo_root, limit)
    print(
        json.dumps(
            {"status": "schema_validated", "tools": validated_tools},
            ensure_ascii=False,
        ),
        flush=True,
    )
    if fresh:
        archived = archive_run(p)
        if archived:
            print(json.dumps({"status": "archived", "output": str(archived)}, ensure_ascii=False))
    generate_trajectories(
        p["questions"],
        p["raw"],
        p["audit"],
        candidate_config,
        model_catalog,
        repo_root,
        p["workspaces"],
        resume=resume,
        limit=limit,
        max_attempts=int(trajectory.get("max_attempts", 2)),
        min_tool_calls=int(trajectory.get("min_tool_calls", 1)),
        min_distinct_tools=int(trajectory.get("min_distinct_tools", 1)),
        max_tool_calls=int(trajectory.get("max_tool_calls", 8)),
        max_tool_result_chars=int(trajectory.get("max_tool_result_chars", 16000)),
        max_concurrency=concurrency,
    )
    print(
        json.dumps(
            {
                "status": "generated",
                "output": str(p["raw"]),
                "max_concurrency": concurrency,
            },
            ensure_ascii=False,
        )
    )


def do_verify(p: dict[str, Path], resume: bool, limit: int | None) -> None:
    verify_trajectories(
        p["raw"],
        p["verified"],
        p["rejected"],
        resume=resume,
        limit=limit,
        model_catalog=_model_catalog(p),
    )
    print(
        json.dumps(
            {"verified": str(p["verified"]), "rejected": str(p["rejected"])},
            ensure_ascii=False,
        )
    )


def do_export(p: dict[str, Path]) -> None:
    summary = export_trajectory_datasets(p["verified"], p["exports"])
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate compact multimodal SFT tool trajectories with one candidate model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("api_sft_single_candidate/config.yaml"),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("generate-trajectories")
    command.add_argument("--limit", type=int)
    command.add_argument("--max-concurrency", type=int)
    mode = command.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--fresh", action="store_true")
    command = sub.add_parser("verify-trajectories")
    command.add_argument("--limit", type=int)
    command.add_argument("--resume", action="store_true")
    sub.add_parser("export-trajectories")
    command = sub.add_parser("run-trajectories")
    command.add_argument("--limit", type=int)
    command.add_argument("--max-concurrency", type=int)
    mode = command.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    config, workspace, p = load(args.config)
    if args.command == "generate-trajectories":
        do_generate(
            config,
            workspace,
            p,
            args.resume,
            args.fresh,
            args.limit,
            args.max_concurrency,
        )
    elif args.command == "verify-trajectories":
        do_verify(p, args.resume, args.limit)
    elif args.command == "export-trajectories":
        do_export(p)
    elif args.command == "run-trajectories":
        do_generate(
            config,
            workspace,
            p,
            args.resume,
            args.fresh,
            args.limit,
            args.max_concurrency,
        )
        do_verify(p, args.resume, args.limit)
        do_export(p)
