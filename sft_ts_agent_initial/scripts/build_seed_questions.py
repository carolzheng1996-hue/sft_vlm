#!/usr/bin/env python3
"""Build SFT seed questions from existing time-series benchmark cases."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_exporter_modules(path: Path) -> list[dict[str, Any]]:
    """Load module-level tool categories from export_data_analysis_tools.py."""
    if not path.exists():
        return []
    spec = importlib.util.spec_from_file_location("export_data_analysis_tools", path)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    records = []
    for item in getattr(module, "DATA_ANALYSIS_MODULES", []):
        records.append(
            {
                "name": Path(str(item.path)).stem,
                "type": item.category,
                "input": "见工具 schema 或源代码定义",
                "output": "工具返回的结构化分析结果或图像 artifact",
                "use_when": f"当任务需要 {item.category} 能力时使用；源模块 {item.path}",
                "source_file": str(item.path),
                "include_by_default": bool(getattr(item, "include_by_default", True)),
            }
        )
    return records


def load_tool_bundle(path: Path) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    if path.suffix == ".jsonl":
        tools = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("record_type") == "tool":
                    tools.append(row)
    else:
        bundle = read_json(path)
        tools = bundle.get("tools", [])
    return [
        {
            "name": t.get("tool_name") or t.get("name") or t.get("factory_function"),
            "type": t.get("category", "tool"),
            "input": t.get("schema_name") or "见工具 schema",
            "output": "结构化工具结果",
            "use_when": t.get("description") or f"使用 {t.get('category', 'tool')} 工具完成对应分析。",
            "permission_name": t.get("permission_name"),
            "source_file": t.get("source_file"),
        }
        for t in tools
        if t.get("tool_name") or t.get("name") or t.get("factory_function")
    ]


def normalize_model_metadata(paths: list[Path]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        data = read_json(path)
        package_name = data.get("name", path.stem)
        runtime_type = data.get("metadata", {}).get("runtime_type") or data.get("metadata", {}).get("catalog_type", "unknown")
        tasks = data.get("tasks") or []
        entries = data.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, str):
                    name = entry
                    model_id = entry
                else:
                    name = entry.get("name") or entry.get("id")
                    model_id = entry.get("id") or name
                if not name:
                    continue
                models.append(
                    {
                        "name": name,
                        "family": runtime_type,
                        "package": package_name,
                        "tasks": tasks,
                        "good_for": f"{package_name} 中的 {name}，适用于 {', '.join(tasks) if tasks else '时序任务'} 的候选模型。",
                        "avoid_when": "数据规模、任务类型、协变量形态、训练/推理预算或解释性要求不匹配时避免盲选。",
                        "id": model_id,
                    }
                )
        for package in data.get("model_packages", []):
            models.append(
                {
                    "name": package.get("name") or package.get("id"),
                    "family": package.get("runtime_type", "model_package"),
                    "package": package_name,
                    "tasks": package.get("tasks", []),
                    "good_for": f"模型包级候选，适用于 {', '.join(package.get('tasks', []))}。",
                    "avoid_when": "需要具体模型能力、输入格式或训练成本判断时，必须下钻到包内模型再选择。",
                    "id": package.get("id"),
                }
            )
    dedup: dict[str, dict[str, Any]] = {}
    for model in models:
        key = f"{model.get('package')}::{model.get('name')}"
        dedup[key] = model
    return list(dedup.values())


def merge_catalog(
    base_catalog: dict[str, Any],
    model_metadata: list[Path],
    tools_source: Path | None,
    tools_bundle: Path | None,
) -> dict[str, Any]:
    catalog = {
        "version": base_catalog.get("version", "merged"),
        "tools": list(base_catalog.get("tools", [])),
        "models": list(base_catalog.get("models", [])),
    }
    real_models = normalize_model_metadata(model_metadata)
    if real_models:
        catalog["models"] = real_models + catalog["models"]
    real_tools = load_tool_bundle(tools_bundle) if tools_bundle else []
    if not real_tools and tools_source:
        real_tools = load_exporter_modules(tools_source)
    if real_tools:
        catalog["tools"] = real_tools + catalog["tools"]
    return catalog


def read_text(path: Path, max_chars: int = 5000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    return text[:max_chars]


def list_cases(cases_root: Path, max_cases: int | None) -> list[Path]:
    cases = sorted(p for p in cases_root.iterdir() if p.is_dir() and p.name.startswith("case_"))
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def existing_files(case_dir: Path, relative_files: list[str]) -> list[str]:
    return [rel for rel in relative_files if (case_dir / rel).exists()]


def compact_case_context(case_dir: Path) -> dict[str, Any]:
    metadata = read_json(case_dir / "metadata.json")
    generation_config = read_json(case_dir / "generation_config.json")
    basic_statistics = read_json(case_dir / "basic_statistics.json")
    data_dictionary = read_text(case_dir / "data_dictionary.md", max_chars=3000)
    return {
        "metadata": metadata,
        "generation_config": generation_config,
        "basic_statistics": basic_statistics,
        "data_dictionary_excerpt": data_dictionary,
    }


def build_system_prompt(catalog: dict[str, Any]) -> str:
    tool_names = ", ".join(t["name"] for t in catalog["tools"][:80])
    model_names = ", ".join(m["name"] for m in catalog["models"][:120])
    return (
        "你是一名资深时序分析与预测建模专家，正在为 VLM/Agent 生成 SFT 答案。"
        "回答必须体现专家工具选择、工具先后关系、哪些问题需要结合图像、哪些可以 text-only 判断。"
        "不要臆造不可见证据；如果只能提出待确认项，要明确说明。"
        f"\n\n可用工具：{tool_names}\n可选模型/策略：{model_names}"
    )


def build_user_prompt(
    case_id: str,
    blueprint: dict[str, Any],
    case_context: dict[str, Any],
    visible_files: list[str],
    catalog: dict[str, Any],
) -> str:
    tool_catalog = [
        {"name": t["name"], "type": t.get("type"), "use_when": t.get("use_when"), "source_file": t.get("source_file")}
        for t in catalog["tools"][:120]
    ]
    model_catalog = [
        {
            "name": m["name"],
            "family": m.get("family"),
            "package": m.get("package"),
            "tasks": m.get("tasks"),
            "good_for": m.get("good_for"),
            "avoid_when": m.get("avoid_when"),
        }
        for m in catalog["models"][:160]
    ]
    payload = {
        "case_id": case_id,
        "task_category": blueprint["category"],
        "input_mode": blueprint["input_mode"],
        "question": blueprint["question"],
        "visible_files": visible_files,
        "case_context": case_context,
        "tool_catalog": tool_catalog,
        "model_catalog": model_catalog,
        "tool_focus": blueprint.get("tool_focus", []),
        "model_focus": blueprint.get("model_focus", []),
        "difficulty": blueprint.get("difficulty", "medium"),
        "expected_trace_hint": blueprint["expected_trace"],
        "answer_must_cover": blueprint["answer_must_cover"],
        "required_output_schema": {
            "tool_plan": [
                {
                    "step": 1,
                    "tool": "tool_name",
                    "why": "为什么在这一步使用",
                    "uses_image": False,
                    "expected_or_observed_result": "基于可见结果的结论或待确认项",
                }
            ],
            "answer": "面向用户的中文专家答案",
            "model_or_strategy_recommendation": ["按优先级排序"],
            "risks_and_uncertainties": ["不能过度断言的地方"],
            "quality_checks": ["用于过滤坏答案的自检项"],
        },
    }
    return (
        "请根据下面 JSON 生成一个高质量 SFT assistant 答案。"
        "答案要自然、简洁但信息密度高；必须输出合法 JSON，不要 Markdown 代码块。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def make_record(case_dir: Path, blueprint: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    case_id = case_dir.name
    text_files = [
        "train.csv",
        "valid.csv",
        "metadata.json",
        "data_dictionary.md",
        "basic_statistics.json",
        "model_metrics.csv",
        "valid_predictions.csv",
    ]
    visible_files = existing_files(case_dir, text_files + blueprint.get("requires_images", []))
    images = existing_files(case_dir, blueprint.get("requires_images", []))
    case_context = compact_case_context(case_dir)
    question_id = f"{case_id}_{blueprint['id']}"
    messages = [
        {"role": "system", "content": build_system_prompt(catalog)},
        {
            "role": "user",
            "content": build_user_prompt(case_id, blueprint, case_context, visible_files, catalog),
        },
    ]
    return {
        "id": question_id,
        "case_id": case_id,
        "category": blueprint["category"],
        "input_mode": blueprint["input_mode"],
        "question": blueprint["question"],
        "messages": messages,
        "images": images,
        "visible_files": visible_files,
        "expert_trace": {
            "expected_tools": blueprint["expected_trace"],
            "answer_must_cover": blueprint["answer_must_cover"],
        },
        "metadata": {
            "case_type": case_context["metadata"].get("case_type"),
            "domain": case_context["metadata"].get("domain"),
            "frequency": case_context["metadata"].get("frequency"),
            "forecast_horizon": case_context["metadata"].get("forecast_horizon"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--model-metadata", type=Path, nargs="*", default=[])
    parser.add_argument("--tools-source", type=Path, default=None)
    parser.add_argument("--tools-bundle", type=Path, default=None)
    parser.add_argument("--blueprints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--input-mode", choices=["all", "text_only", "image_text"], default="all")
    args = parser.parse_args()

    catalog = merge_catalog(read_json(args.catalog), args.model_metadata, args.tools_source, args.tools_bundle)
    blueprints = read_json(args.blueprints)
    if args.input_mode != "all":
        blueprints = [b for b in blueprints if b["input_mode"] == args.input_mode]

    records = []
    for case_dir in list_cases(args.cases_root, args.max_cases):
        required = ["metadata.json", "generation_config.json", "basic_statistics.json"]
        if not all((case_dir / name).exists() for name in required):
            continue
        for blueprint in blueprints:
            records.append(make_record(case_dir, blueprint, catalog))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "output": str(args.out),
        "num_records": len(records),
        "num_cases": len({r["case_id"] for r in records}),
        "category_counts": {},
        "input_mode_counts": {},
    }
    for record in records:
        summary["category_counts"][record["category"]] = summary["category_counts"].get(record["category"], 0) + 1
        summary["input_mode_counts"][record["input_mode"]] = summary["input_mode_counts"].get(record["input_mode"], 0) + 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
