from __future__ import annotations

from pathlib import Path
from typing import Any
import re

from .common import read_json, sha256_file, write_json


def normalize_tools(bundle_path: Path) -> dict[str, Any]:
    bundle = read_json(bundle_path)
    tools=[]
    for item in bundle.get("tools", []):
        source=item.get("implementation_source","")
        argument_keys=sorted(set(re.findall(r"args(?:\.get\(|\[)[\"']([^\"']+)[\"']",source)))
        tools.append({"name":item.get("tool_name"),"permission_name":item.get("permission_name"),"category":item.get("category"),"description":item.get("description"),"schema_name":item.get("schema_name"),"argument_keys":argument_keys,"source_file":item.get("source_file"),"implementation_sha256":item.get("implementation_sha256")})
    if not tools or any(not item["name"] for item in tools):
        raise ValueError(f"Invalid data analysis tools bundle: {bundle_path}")
    return {
        "source": str(bundle_path),
        "source_sha256": sha256_file(bundle_path),
        "mcp_server_name": bundle.get("metadata", {}).get("mcp_server_name", "tsa"),
        "tool_count": len(tools),
        "tools": tools,
    }


def normalize_models(paths: list[Path]) -> dict[str, Any]:
    found: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = read_json(path)
        package = data.get("name", path.stem)
        runtime = data.get("metadata", {}).get("runtime_type", data.get("metadata", {}).get("catalog_type", "unknown"))
        tasks = data.get("tasks", [])
        for entry in data.get("entries", []):
            item = {"id": entry, "name": entry} if isinstance(entry, str) else entry
            name = item.get("name") or item.get("id")
            if name:
                found[f"{package}::{name}"] = {"name": name, "id": item.get("id", name), "package": package, "runtime_type": runtime, "tasks": tasks}
        for item in data.get("model_packages", []):
            name = item.get("name") or item.get("id")
            if name:
                found[f"{package}::{name}"] = {"name": name, "id": item.get("id", name), "package": package, "runtime_type": item.get("runtime_type", "package"), "tasks": item.get("tasks", [])}
    return {"sources": [str(path) for path in paths], "model_count": len(found), "models": list(found.values())}


def prepare_catalogs(bundle_path: Path, metadata_paths: list[Path], output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    tools = normalize_tools(bundle_path)
    models = normalize_models(metadata_paths)
    write_json(output_dir / "tools.normalized.json", tools)
    write_json(output_dir / "models.normalized.json", models)
    return tools, models


def prepare_catalogs_from_repo(repo_root: Path, metadata_paths: list[Path], output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze the current claude_tsa data-analysis source before normalizing it."""
    from agent_tools.export_data_analysis_tools import build_bundle, select_modules, write_bundle

    bundle = build_bundle(
        repo_root.resolve(),
        select_modules(include_training=False, include_knowledge=False, all_tsa_tools=False),
    )
    bundle_path = output_dir / "data_analysis_tools_bundle.json"
    write_bundle(bundle, bundle_path, "json")
    return prepare_catalogs(bundle_path, metadata_paths, output_dir)
