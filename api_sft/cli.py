from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .answers import generate_answers, select_answers
from .catalogs import prepare_catalogs, prepare_catalogs_from_repo
from .common import iter_jsonl, load_env_file, load_yaml, read_json, resolve_path, write_json, write_jsonl
from .exporters import export_datasets
from .questions import (
    QUESTION_AUDIT_FORMAT_VERSION,
    QUESTION_CONTRACT_VERSION,
    QUESTION_RUNTIME_FORMAT_VERSION,
    QUESTION_WRITER_PROMPT_ID,
    build_question_specs,
    generate_questions,
    refresh_question_records,
    rewrite_questions,
)
from .verify import verify_records
from .trajectories_compact import COMPETITION_AUDIT_FORMAT_VERSION, TRAJECTORY_FORMAT_VERSION, generate_trajectories
from .trajectory_exporters_compact import export_trajectory_datasets
from .trajectory_verify_compact import verify_trajectories


def paths(config: dict[str,Any], workspace: Path) -> dict[str,Path]:
    root=resolve_path(config.get("output_dir","api_sft/output"),workspace)
    scenario_dir=root/str(config.get("scenario_generation",{}).get("output_subdir","scenarios"))
    return {"root":root,"catalogs":root/"catalogs","scenario_dir":scenario_dir,"scenarios":scenario_dir/"scenarios.jsonl","source_legacy_scenarios":root/"legacy"/"scenarios_v3"/"scenarios.jsonl","question_specs":root/"question_specs.jsonl","questions":root/"questions.final.jsonl","question_audit":root/"questions.audit.jsonl","question_rejected":root/"questions.rejected.jsonl","question_archives":root/"question_run_archives","candidates":root/"answer_candidates.jsonl","selected":root/"selected_answers.jsonl","verified":root/"verified_answers.jsonl","rejected":root/"rejected.jsonl","first_rejected":root/"rejected.first_pass.jsonl","retry_questions":root/"questions.retry.jsonl","retry_candidates":root/"answer_candidates.retry.jsonl","retry_selected":root/"selected_answers.retry.jsonl","coverage":root/"coverage_report.json","exports":root/"exports","trajectory_workspaces":root/"trajectory_workspaces","trajectories":root/"trajectories.raw.jsonl","trajectory_audit":root/"trajectories.competition_audit.jsonl","trajectories_verified":root/"trajectories.verified.jsonl","trajectories_rejected":root/"trajectories.rejected.jsonl","trajectory_exports":root/"trajectory_exports","trajectory_archives":root/"trajectory_run_archives"}


def load(config_path: Path) -> tuple[dict[str,Any],Path,dict[str,Path]]:
    config=load_yaml(config_path.resolve()); workspace=Path.cwd().resolve(); env_file=config.get("env_file")
    if env_file: load_env_file(resolve_path(env_file,workspace),bool(config.get("env_override",False)))
    return config,workspace,paths(config,workspace)


def do_catalogs(config: dict[str,Any], workspace: Path, p: dict[str,Path]) -> None:
    catalog=config["catalogs"]; metadata=[resolve_path(x,workspace) for x in catalog["model_metadata"]]
    repo_value=catalog.get("tools_repo_root")
    if repo_value:
        repo=resolve_path(repo_value,workspace); tools,models=prepare_catalogs_from_repo(repo,metadata,p["catalogs"])
    else:
        bundle=resolve_path(catalog["tools_bundle"],workspace); tools,models=prepare_catalogs(bundle,metadata,p["catalogs"])
    print(json.dumps({"tools":tools["tool_count"],"models":models["model_count"],"output":str(p["catalogs"])},ensure_ascii=False))


def _question_output_paths(p: dict[str, Path]) -> list[Path]:
    return [p["questions"], p["question_audit"], p["question_rejected"], p["coverage"]]


def archive_question_run(p: dict[str, Path], reason: str = "fresh_restart") -> Path | None:
    existing = [path for path in _question_output_paths(p) if path.exists()]
    if not existing:
        return None
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f%z")
    archive_dir = p["question_archives"] / timestamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []
    moved: list[tuple[Path, Path]] = []
    try:
        for source in existing:
            line_count = sum(1 for _ in iter_jsonl(source)) if source.suffix == ".jsonl" else None
            target = archive_dir / source.name
            shutil.move(str(source), str(target))
            moved.append((source, target))
            entries.append({"source": str(source), "archived_as": source.name, "jsonl_records": line_count})
    except Exception:
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                shutil.move(str(target), str(source))
        raise
    write_json(archive_dir / "archive_manifest.json", {
        "archived_at": datetime.now().astimezone().isoformat(),
        "status": "stale_archived",
        "reason": reason,
        "entries": entries,
    })
    return archive_dir


def _assert_resumable_question_formats(p: dict[str, Path]) -> None:
    final_path = p["questions"]
    audit_path = p["question_audit"]
    if not final_path.exists() and not audit_path.exists():
        return
    if not final_path.exists() or not audit_path.exists():
        raise RuntimeError("cannot resume questions without both final and audit; use --fresh to archive and restart")
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
        raise RuntimeError(
            "cannot resume stale question protocol; use --fresh to archive the old questions and restart"
        )


def _trajectory_output_paths(p: dict[str, Path]) -> list[Path]:
    return [
        p["trajectories"],
        p["trajectory_audit"],
        p["trajectories_verified"],
        p["trajectories_rejected"],
        p["trajectory_workspaces"],
        p["trajectory_exports"],
    ]


def archive_trajectory_run(p: dict[str, Path], reason: str = "fresh_restart") -> Path | None:
    existing = [path for path in _trajectory_output_paths(p) if path.exists()]
    if not existing:
        return None
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f%z")
    archive_dir = p["trajectory_archives"] / timestamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []
    for source in existing:
        if source.is_file():
            observed_versions: set[str] = set()
            if source.suffix == ".jsonl":
                line_count = 0
                with source.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        line_count += 1
                        value = json.loads(line)
                        if value.get("format_version"):
                            observed_versions.add(str(value["format_version"]))
            else:
                line_count = None
            file_count = 1
        else:
            line_count = None
            file_count = sum(1 for item in source.rglob("*") if item.is_file())
            observed_versions = set()
        entries.append({"source": str(source), "archived_as": source.name, "file_count": file_count, "jsonl_records": line_count, "observed_format_versions": sorted(observed_versions)})
    moved: list[tuple[Path, Path]] = []
    try:
        for source in existing:
            target = archive_dir / source.name
            shutil.move(str(source), str(target))
            moved.append((source, target))
    except Exception:
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                shutil.move(str(target), str(source))
        raise
    write_json(archive_dir / "archive_manifest.json", {
        "archived_at": datetime.now().astimezone().isoformat(),
        "status": "stale_archived",
        "reason": reason,
        "trajectory_format_version": TRAJECTORY_FORMAT_VERSION,
        "competition_audit_format_version": COMPETITION_AUDIT_FORMAT_VERSION,
        "entries": entries,
    })
    return archive_dir


def _assert_resumable_trajectory_formats(p: dict[str, Path]) -> None:
    expected = {
        p["trajectories"]: TRAJECTORY_FORMAT_VERSION,
        p["trajectory_audit"]: COMPETITION_AUDIT_FORMAT_VERSION,
    }
    for path, expected_version in expected.items():
        if not path.exists():
            continue
        observed = {str(row.get("format_version")) for row in iter_jsonl(path)}
        if observed - {expected_version}:
            raise RuntimeError(
                f"cannot resume mixed or stale trajectory format in {path}: "
                f"expected {expected_version}, observed {sorted(observed)}; use --fresh to archive and restart"
            )


def do_generate_trajectories(config: dict[str,Any], workspace: Path, p: dict[str,Path], resume: bool=False, limit: int|None=None, fresh: bool=False) -> None:
    if resume and fresh:
        raise RuntimeError("--resume and --fresh are mutually exclusive")
    existing_outputs = [path for path in _trajectory_output_paths(p) if path.exists()]
    if resume:
        _assert_resumable_trajectory_formats(p)
    elif not fresh and existing_outputs:
        raise RuntimeError("trajectory outputs already exist; use --resume to continue or --fresh to archive them and start clean")
    trajectory=config.get("trajectory_generation",{}); catalog=config["catalogs"]
    repo_value=catalog.get("tools_repo_root")
    if not repo_value: raise RuntimeError("catalogs.tools_repo_root is required for live tool trajectory generation")
    candidates=config.get("models",{}).get("trajectory_candidates")
    selector=config.get("models",{}).get("trajectory_selector")
    if not isinstance(candidates,list) or len(candidates)!=2: raise RuntimeError("models.trajectory_candidates must contain exactly two models")
    if not isinstance(selector,dict): raise RuntimeError("models.trajectory_selector is required for trajectory competition")
    candidate_configs=[]
    for model in candidates:
        if ".example" in str(model.get("base_url", "")) or "replace-with" in str(model.get("model", "")):
            raise RuntimeError("trajectory_candidates still contains a placeholder endpoint/model; configure two real tool-call-capable models before the pilot")
        model_config=dict(model); model_config.setdefault("timeout_seconds",int(trajectory.get("timeout_seconds",180))); model_config.setdefault("retries",int(trajectory.get("retries",3))); model_config["response_format"]=False; model_config["parallel_tool_calls"]=False; candidate_configs.append(model_config)
    selector_config=dict(selector); selector_config.setdefault("timeout_seconds",int(trajectory.get("selector_timeout_seconds",trajectory.get("timeout_seconds",180)))); selector_config.setdefault("retries",int(trajectory.get("retries",3))); selector_config["response_format"]=True
    model_catalog=read_json(p["catalogs"]/"models.normalized.json")["models"]
    if fresh:
        archived = archive_trajectory_run(p)
        if archived:
            print(json.dumps({"status":"archived_previous_trajectory_run","output":str(archived)},ensure_ascii=False))
    generate_trajectories(
        p["questions"],p["trajectories"],p["trajectory_audit"],candidate_configs,selector_config,model_catalog,resolve_path(repo_value,workspace),p["trajectory_workspaces"],resume,limit,
        int(trajectory.get("max_attempts",2)),int(trajectory.get("min_tool_calls",1)),int(trajectory.get("min_distinct_tools",1)),int(trajectory.get("max_tool_calls",8)),int(trajectory.get("max_tool_result_chars",16000)),
    )
    print(json.dumps({"status":"generated","output":str(p["trajectories"])},ensure_ascii=False))


def do_verify_trajectories(p: dict[str,Path], resume: bool=False, limit: int|None=None) -> None:
    model_catalog=read_json(p["catalogs"]/"models.normalized.json")["models"]
    verify_trajectories(p["trajectories"],p["trajectories_verified"],p["trajectories_rejected"],resume,limit,model_catalog)
    print(json.dumps({"verified":str(p["trajectories_verified"]),"rejected":str(p["trajectories_rejected"])},ensure_ascii=False))


def do_scenarios(config: dict[str,Any], p: dict[str,Path], workspace: Path, count: int|None=None, limit: int|None=None, resume: bool=False) -> None:
    from .scenarios import GENERATOR_VERSION, generate_scenarios

    amount=count or int(config.get("scenario_count",200))
    existing_count=sum(1 for _ in iter_jsonl(p["scenarios"])) if resume and p["scenarios"].exists() else 0
    amount=max(amount,existing_count)
    if limit and not existing_count: amount=min(amount,limit)
    matrix=load_yaml(resolve_path(config.get("coverage_matrix","api_sft/coverage_matrix.yaml"),workspace)); ratios=matrix["target_distribution"]["task"]
    rows=generate_scenarios(amount,int(config.get("seed",20260712)),p["scenario_dir"],resume,ratios,config.get("scenario_generation")); complexities={name:sum(row.get("signal_complexity")==name for row in rows) for name in ["controlled","compositional","confounded"]}; print(json.dumps({"scenarios":len(rows),"generator_version":GENERATOR_VERSION,"complexity_counts":complexities,"output":str(p["scenarios"])},ensure_ascii=False))


def do_migrate_scenarios(p: dict[str,Path], limit: int|None=None, resume: bool=False) -> None:
    from .scenarios import GENERATOR_VERSION, migrate_scenarios

    if not p["source_legacy_scenarios"].exists(): raise FileNotFoundError(f"Legacy scenario manifest not found: {p['source_legacy_scenarios']}")
    rows=migrate_scenarios(p["source_legacy_scenarios"],p["scenario_dir"],limit,resume)
    print(json.dumps({"scenarios":len(rows),"generator_version":GENERATOR_VERSION,"data_layout":"wide_panel_v1","output":str(p["scenarios"])},ensure_ascii=False))


def do_question_specs(config: dict[str,Any], p: dict[str,Path], workspace: Path, limit: int|None=None, resume: bool=False) -> None:
    task_pool=resolve_path(config.get("task_pool","api_sft/task_pool.yaml"),workspace)
    target_prompt=resolve_path("api_sft/prompts/tool_execution_system.txt",workspace)
    if not target_prompt.exists(): raise FileNotFoundError(f"Target tool-execution system prompt not found: {target_prompt}")
    rows=build_question_specs(p["scenarios"],p["catalogs"]/"tools.normalized.json",p["catalogs"]/"models.normalized.json",p["question_specs"],int(config.get("questions_per_scenario",2)),limit,resume,task_pool,config.get("modality_sampling"),int(config.get("seed",20260712)))
    print(json.dumps({"question_specs":len(rows),"output":str(p["question_specs"])},ensure_ascii=False))


def do_questions(config: dict[str,Any], p: dict[str,Path], workspace: Path, model_config: dict[str,Any], limit: int|None=None, resume: bool=False, fresh: bool=False) -> None:
    if resume and fresh: raise RuntimeError("--resume and --fresh are mutually exclusive")
    existing_outputs=[path for path in _question_output_paths(p) if path.exists()]
    if resume:
        _assert_resumable_question_formats(p)
    elif not fresh and existing_outputs:
        raise RuntimeError("question outputs already exist; use --resume to continue or --fresh to archive them and start clean")
    if fresh:
        archived=archive_question_run(p)
        if archived: print(json.dumps({"status":"archived_previous_question_run","output":str(archived)},ensure_ascii=False))
    generation=config.get("generation",{}); writer_path=resolve_path(generation.get("question_writer_prompt","api_sft/prompts/question_writer_system.txt"),workspace)
    cfg=dict(model_config); cfg.setdefault("timeout_seconds",int(generation.get("timeout_seconds",120))); cfg.setdefault("retries",int(generation.get("retries",3)))
    rows=generate_questions(p["question_specs"],p["questions"],p["question_rejected"],cfg,writer_path,resume,limit,int(generation.get("question_max_attempts",2)),p["question_audit"])
    print(json.dumps({"questions":len(rows),"output":str(p["questions"]),"audit":str(p["question_audit"]),"rejected":str(p["question_rejected"]),"coverage":str(p["coverage"])},ensure_ascii=False))


def do_verify(config: dict[str,Any], p: dict[str,Path], models: dict[str,Any], timeout: int, retries: int, concurrency: int, offline: bool, resume: bool, limit: int|None, retry_low_quality: bool=True) -> None:
    coverage=json.loads(p["coverage"].read_text(encoding="utf-8"))
    if not coverage.get("passed"): raise SystemExit(f"Coverage validation failed: {coverage.get('missing_required_values')}")
    use_retry=retry_low_quality and not offline; first_target=p["first_rejected"] if use_retry else p["rejected"]
    if use_retry and not resume and p["rejected"].exists(): p["rejected"].unlink()
    verify_records(p["selected"],p["scenarios"],p["catalogs"]/"tools.normalized.json",p["catalogs"]/"models.normalized.json",p["verified"],first_target,float(config.get("minimum_quality_score",.75)),None if offline else models.get("verifier"),timeout,retries,resume,limit)
    failures=list(iter_jsonl(first_target)) if use_retry and first_target.exists() else []
    if not failures: return
    retry_questions=[]
    for failed in failures:
        q=dict(failed["question_record"]); flags=failed.get("deterministic_review",{}).get("flags",[])
        q["retry_feedback"]={"attempt":2,"instruction":"上一版未通过质量门槛。只能依据原有可见文本和图像重新回答；降低无证据断言，补全工具依赖、证据边界、验证与风险。","deterministic_flags":flags}
        retry_questions.append(q)
    write_jsonl(p["retry_questions"],retry_questions)
    answer_models=models.get("answer_vlms") or models.get("trajectory_candidates") or []
    selector=models.get("selector") or models.get("trajectory_selector")
    generate_answers(p["retry_questions"],p["retry_candidates"],answer_models,concurrency,timeout,retries,False,None)
    select_answers(p["retry_candidates"],p["retry_selected"],selector,timeout,retries,False,None)
    verify_records(p["retry_selected"],p["scenarios"],p["catalogs"]/"tools.normalized.json",p["catalogs"]/"models.normalized.json",p["verified"],p["rejected"],float(config.get("minimum_quality_score",.75)),models.get("verifier"),timeout,retries,True,None)


def main() -> None:
    parser=argparse.ArgumentParser(description="Build multimodal time-series SFT data with configurable VLM APIs")
    parser.add_argument("--config",type=Path,default=Path("api_sft/config.example.yaml")); sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("prepare-catalogs")
    s=sub.add_parser("generate-scenarios"); s.add_argument("--count",type=int); s.add_argument("--limit",type=int); s.add_argument("--resume",action="store_true")
    s=sub.add_parser("migrate-scenarios"); s.add_argument("--limit",type=int); s.add_argument("--resume",action="store_true")
    s=sub.add_parser("generate-question-specs"); s.add_argument("--limit",type=int); s.add_argument("--resume",action="store_true")
    s=sub.add_parser("generate-questions"); s.add_argument("--limit",type=int); g=s.add_mutually_exclusive_group(); g.add_argument("--resume",action="store_true"); g.add_argument("--fresh",action="store_true")
    sub.add_parser("refresh-question-records")
    for name in ["rewrite-questions","generate-answers","select-answers"]:
        s=sub.add_parser(name); s.add_argument("--limit",type=int); s.add_argument("--resume",action="store_true")
    s=sub.add_parser("verify"); s.add_argument("--limit",type=int); s.add_argument("--resume",action="store_true"); s.add_argument("--offline",action="store_true",help="Run deterministic checks without API verification or retry"); s.add_argument("--no-retry",action="store_true")
    sub.add_parser("export")
    s=sub.add_parser("generate-trajectories"); s.add_argument("--limit",type=int); g=s.add_mutually_exclusive_group(); g.add_argument("--resume",action="store_true"); g.add_argument("--fresh",action="store_true")
    s=sub.add_parser("verify-trajectories"); s.add_argument("--limit",type=int); s.add_argument("--resume",action="store_true")
    sub.add_parser("export-trajectories")
    s=sub.add_parser("run-trajectories"); s.add_argument("--limit",type=int); g=s.add_mutually_exclusive_group(); g.add_argument("--resume",action="store_true"); g.add_argument("--fresh",action="store_true")
    s=sub.add_parser("run-all"); s.add_argument("--limit",type=int); s.add_argument("--resume",action="store_true")
    args=parser.parse_args(); config,workspace,p=load(args.config); generation=config.get("generation",{}); timeout=int(generation.get("timeout_seconds",120)); retries=int(generation.get("retries",3)); concurrency=int(generation.get("max_concurrency",4)); models=config.get("models",{})
    if args.command=="prepare-catalogs": do_catalogs(config,workspace,p)
    elif args.command=="generate-scenarios": do_scenarios(config,p,workspace,args.count,args.limit,args.resume)
    elif args.command=="migrate-scenarios": do_migrate_scenarios(p,args.limit,args.resume)
    elif args.command=="generate-question-specs": do_question_specs(config,p,workspace,args.limit,args.resume)
    elif args.command=="generate-questions": do_questions(config,p,workspace,models["question"],args.limit,args.resume,args.fresh)
    elif args.command=="refresh-question-records":
        rows=refresh_question_records(p["question_specs"],p["questions"],p["question_audit"]); print(json.dumps({"questions":len(rows),"output":str(p["questions"]),"audit":str(p["question_audit"]),"api_calls":0},ensure_ascii=False))
    elif args.command=="rewrite-questions":
        print("WARNING: rewrite-questions 已弃用；当前命令等价于 generate-questions。")
        generation=config.get("generation",{}); writer_path=resolve_path(generation.get("question_writer_prompt","api_sft/prompts/question_writer_system.txt"),workspace)
        rewrite_questions(p["question_specs"],p["questions"],p["question_rejected"],models["question"],writer_path,args.resume,args.limit,int(generation.get("question_max_attempts",2)),p["question_audit"])
    elif args.command=="generate-answers":
        try: generate_answers(p["questions"],p["candidates"],models.get("answer_vlms") or models.get("trajectory_candidates") or [],concurrency,timeout,retries,args.resume,args.limit)
        except RuntimeError as exc: raise SystemExit(str(exc)) from None
    elif args.command=="select-answers": select_answers(p["candidates"],p["selected"],models.get("selector") or models.get("trajectory_selector"),timeout,retries,args.resume,args.limit)
    elif args.command=="verify":
        do_verify(config,p,models,timeout,retries,concurrency,args.offline,args.resume,args.limit,not args.no_retry)
    elif args.command=="export": print(json.dumps(export_datasets(p["verified"],p["exports"],p["rejected"],p["coverage"]),ensure_ascii=False))
    elif args.command=="generate-trajectories": do_generate_trajectories(config,workspace,p,args.resume,args.limit,args.fresh)
    elif args.command=="verify-trajectories": do_verify_trajectories(p,args.resume,args.limit)
    elif args.command=="export-trajectories": print(json.dumps(export_trajectory_datasets(p["trajectories_verified"],p["trajectory_exports"]),ensure_ascii=False))
    elif args.command=="run-trajectories":
        do_generate_trajectories(config,workspace,p,args.resume,args.limit,args.fresh); do_verify_trajectories(p,args.resume,args.limit); print(json.dumps(export_trajectory_datasets(p["trajectories_verified"],p["trajectory_exports"]),ensure_ascii=False))
    elif args.command=="run-all":
        do_catalogs(config,workspace,p); do_scenarios(config,p,workspace,limit=args.limit,resume=args.resume); do_question_specs(config,p,workspace,args.limit,args.resume); do_questions(config,p,workspace,models["question"],args.limit,args.resume)
        print(json.dumps({"status":"questions_generated","next_stage":"run-trajectories","output":str(p["questions"])},ensure_ascii=False))


if __name__=="__main__": main()
