#!/usr/bin/env python
"""Export data-analysis MCP tool definitions and source code.

The generated bundle is intended as source material for building tool-call
datasets. It includes tool names, permission names, decorator metadata, each
tool factory implementation, and the source files that define the tools and
their shared schemas/helpers.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MCP_SERVER_NAME = "tsa"
DEFAULT_OUTPUT = Path("tool_dataset_sources/data_analysis_tools_bundle.json")


@dataclass(frozen=True)
class ToolModule:
    """A module that contains first-party MCP tool factories."""

    category: str
    path: Path
    include_by_default: bool = True


DATA_ANALYSIS_MODULES = [
    ToolModule("data_io", Path("pulsar/tsa/tools/data_io.py")),
    ToolModule("visualization", Path("pulsar/tsa/tools/visualization.py")),
    ToolModule("visualization", Path("pulsar/tsa/tools/visualization_tools.py")),
    ToolModule("reporting", Path("pulsar/tsa/tools/reporting.py")),
    ToolModule("artifacts", Path("pulsar/tsa/tools/artifacts.py")),
    ToolModule("channel", Path("pulsar/tsa/tools/channel.py")),
    ToolModule("basic_time_series_analysis", Path("pulsar/tsa/tools/tsa_analysis.py")),
    ToolModule("statistics", Path("pulsar/tsa/tools/stats_tools.py")),
    ToolModule("series_analysis", Path("pulsar/tsa/tools/series_analysis.py")),
    ToolModule("trend_detection", Path("pulsar/tsa/tools/trend_detection.py")),
    ToolModule("feature_engineering", Path("pulsar/tsa/tools/feature_extraction.py")),
    ToolModule("feature_engineering", Path("pulsar/tsa/tools/feature_generation.py")),
    ToolModule("causal_analysis", Path("pulsar/tsa/tools/causal_analysis.py")),
    ToolModule("training_jobs", Path("pulsar/tsa/tools/training_jobs.py"), include_by_default=False),
    ToolModule("knowledge_retrieval", Path("pulsar/tsa/tools/knowledge.py"), include_by_default=False),
]


SUPPORTING_SOURCE_FILES = [
    Path("pulsar/tsa/tools/registry.py"),
    Path("pulsar/tsa/tools/schemas.py"),
    Path("pulsar/tsa/tools/io_utils.py"),
    Path("pulsar/tsa/tools/models.py"),
    Path("pulsar/tsa/tools/__init__.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of this script directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output file path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl", "markdown"],
        default="json",
        help="Bundle format.",
    )
    parser.add_argument(
        "--include-training",
        action="store_true",
        help="Include model_train/model_status/model_cancel/model_list_jobs tools.",
    )
    parser.add_argument(
        "--include-knowledge",
        action="store_true",
        help="Include knowledge_search in addition to data-analysis tools.",
    )
    parser.add_argument(
        "--all-tsa-tools",
        action="store_true",
        help="Include every first-party TSA MCP tool module known to this exporter.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    modules = select_modules(
        include_training=args.include_training,
        include_knowledge=args.include_knowledge,
        all_tsa_tools=args.all_tsa_tools,
    )
    bundle = build_bundle(repo_root, modules)
    write_bundle(bundle, (repo_root / args.output).resolve(), args.format)


def select_modules(
    *,
    include_training: bool,
    include_knowledge: bool,
    all_tsa_tools: bool,
) -> list[ToolModule]:
    selected: list[ToolModule] = []
    for module in DATA_ANALYSIS_MODULES:
        if all_tsa_tools or module.include_by_default:
            selected.append(module)
            continue
        if module.category == "training_jobs" and include_training:
            selected.append(module)
        elif module.category == "knowledge_retrieval" and include_knowledge:
            selected.append(module)
    return selected


def build_bundle(repo_root: Path, modules: list[ToolModule]) -> dict[str, Any]:
    tool_records: list[dict[str, Any]] = []
    module_sources: list[dict[str, Any]] = []

    for module in modules:
        source_path = repo_root / module.path
        source = read_text(source_path)
        module_sources.append(source_record(repo_root, source_path, source, module.category))
        tool_records.extend(extract_tools_from_module(repo_root, source_path, source, module.category))

    supporting_sources = []
    for relative_path in SUPPORTING_SOURCE_FILES:
        source_path = repo_root / relative_path
        if source_path.exists():
            supporting_sources.append(
                source_record(repo_root, source_path, read_text(source_path), "supporting_code")
            )

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(repo_root),
            "mcp_server_name": MCP_SERVER_NAME,
            "tool_count": len(tool_records),
            "module_count": len(module_sources),
            "purpose": "Source bundle for constructing MCP tool-call datasets.",
        },
        "tools": tool_records,
        "module_sources": module_sources,
        "supporting_sources": supporting_sources,
    }


def extract_tools_from_module(
    repo_root: Path,
    source_path: Path,
    source: str,
    category: str,
) -> list[dict[str, Any]]:
    tree = ast.parse(source, filename=str(source_path))
    constants = module_constants(tree)
    records: list[dict[str, Any]] = []

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorated_handler = first_decorated_handler(node)
        if decorated_handler is None:
            continue
        handler_node, decorator = decorated_handler

        tool_name_expr = decorator.args[0] if decorator.args else None
        tool_name = resolve_string(tool_name_expr, constants)
        if tool_name is None:
            tool_name = node.name

        description = resolve_string(decorator.args[1], constants) if len(decorator.args) > 1 else None
        schema_name = resolve_name(decorator.args[2]) if len(decorator.args) > 2 else None

        implementation = ast.get_source_segment(source, node) or ""
        records.append(
            {
                "tool_name": tool_name,
                "permission_name": f"mcp__{MCP_SERVER_NAME}__{tool_name}",
                "factory_function": node.name,
                "handler_function": handler_node.name,
                "category": category,
                "source_file": str(source_path.relative_to(repo_root)),
                "description": description,
                "schema_name": schema_name,
                "start_line": node.lineno,
                "end_line": getattr(node, "end_lineno", None),
                "implementation_sha256": sha256_text(implementation),
                "implementation_source": implementation,
            }
        )

    return records


def first_decorated_handler(
    factory_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Call] | None:
    for node in ast.walk(factory_node):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and resolve_name(decorator.func) == "tool":
                return node, decorator
    return None


def module_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = node.value.value
    return constants


def resolve_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = resolve_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def resolve_string(node: ast.AST | None, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    name = resolve_name(node)
    if name is not None:
        return constants.get(name)
    return None


def source_record(repo_root: Path, source_path: Path, source: str, category: str) -> dict[str, Any]:
    return {
        "path": str(source_path.relative_to(repo_root)),
        "category": category,
        "sha256": sha256_text(source),
        "source_code": source,
    }


def write_bundle(bundle: dict[str, Any], output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif output_format == "jsonl":
        rows = [
            *({"record_type": "tool", **tool} for tool in bundle["tools"]),
            *({"record_type": "module_source", **source} for source in bundle["module_sources"]),
            *({"record_type": "supporting_source", **source} for source in bundle["supporting_sources"]),
        ]
        output_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    elif output_format == "markdown":
        output_path.write_text(to_markdown(bundle), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    print(f"Wrote {len(bundle['tools'])} tools to {output_path}")


def to_markdown(bundle: dict[str, Any]) -> str:
    lines = [
        "# Data Analysis Tool Source Bundle",
        "",
        f"- Generated at: `{bundle['metadata']['generated_at']}`",
        f"- MCP server: `{bundle['metadata']['mcp_server_name']}`",
        f"- Tool count: `{bundle['metadata']['tool_count']}`",
        "",
        "## Tools",
        "",
    ]
    for tool in bundle["tools"]:
        lines.extend(
            [
                f"### `{tool['permission_name']}`",
                "",
                f"- Category: `{tool['category']}`",
                f"- Factory: `{tool['factory_function']}`",
                f"- Handler: `{tool['handler_function']}`",
                f"- Source: `{tool['source_file']}:{tool['start_line']}`",
                f"- Schema: `{tool['schema_name']}`",
                f"- Description: {tool['description'] or ''}",
                "",
                "```python",
                tool["implementation_source"],
                "```",
                "",
            ]
        )

    lines.extend(["## Module Sources", ""])
    for source in [*bundle["module_sources"], *bundle["supporting_sources"]]:
        lines.extend(
            [
                f"### `{source['path']}`",
                "",
                "```python",
                source["source_code"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
