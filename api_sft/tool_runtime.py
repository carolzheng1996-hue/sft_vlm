from __future__ import annotations

import asyncio
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import sha256_file, stable_hash


NON_DATA_ANALYSIS_TOOLS = {
    "knowledge_search",
    "model_cancel",
    "model_list_jobs",
    "model_status",
    "model_train",
    "model_training_preflight",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _jsonable(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            default=lambda item: item.item() if hasattr(item, "item") else str(item),
        )
    )


class ClaudeTsaToolRuntime:
    """Execute the existing claude_tsa SDK MCP handlers in an isolated session."""

    def __init__(self, repo_root: Path, workspace_root: Path):
        if sys.version_info < (3, 11):
            raise RuntimeError(
                "claude_tsa requires Python 3.11 or newer. Run the trajectory commands with "
                "the configured claude_tsa Python environment."
            )
        self.repo_root = repo_root.expanduser().resolve()
        self.workspace_root = workspace_root.expanduser().resolve()
        if not (self.repo_root / "pulsar" / "tsa" / "tools").is_dir():
            raise FileNotFoundError(f"Invalid claude_tsa repository: {self.repo_root}")
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        from pulsar.tsa.artifacts import ArtifactStore
        from pulsar.tsa.sessions import SessionManager
        from pulsar.tsa.tools import ToolRegistry

        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.session_manager = SessionManager(self.workspace_root)
        self.artifact_store = ArtifactStore(self.workspace_root)
        self.registry = ToolRegistry(self.workspace_root, artifact_store=self.artifact_store, session_manager=self.session_manager)
        self.tool_map = {tool.name: tool for tool in self.registry.tools()}
        self.session = None
        self.allowed_names: set[str] = set()
        self.dataset_uri = "uploads/dataset.csv"

    def source_snapshot(self) -> dict[str, Any]:
        head = None
        dirty = None
        try:
            head_result = subprocess.run(
                ["git", "-C", str(self.repo_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            status_result = subprocess.run(
                ["git", "-C", str(self.repo_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            )
            head = head_result.stdout.strip()
            dirty = bool(status_result.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
        return {"repo_root": str(self.repo_root), "git_head": head, "git_dirty": dirty}

    def describe_tools(self, names: set[str] | None = None, session_id: str | None = None) -> list[dict[str, Any]]:
        selected = names or (set(self.tool_map) - NON_DATA_ANALYSIS_TOOLS)
        result: list[dict[str, Any]] = []
        for name in sorted(selected):
            tool = self.tool_map.get(name)
            if tool is None:
                continue
            schema = _jsonable(tool.input_schema)
            if session_id:
                properties = schema.setdefault("properties", {})
                session_schema = dict(properties.get("session_id") or {"type": "string"})
                session_schema["const"] = session_id
                properties["session_id"] = session_schema
                required = list(schema.get("required") or [])
                if "session_id" not in required:
                    required.append("session_id")
                schema["required"] = required
            source_file = inspect.getsourcefile(tool.handler)
            source_path = Path(source_file).resolve() if source_file else None
            result.append(
                {
                    "name": name,
                    "permission_name": f"mcp__tsa__{name}",
                    "description": tool.description,
                    "input_schema": schema,
                    "argument_keys": sorted(schema.get("properties", {})),
                    "source_file": str(source_path.relative_to(self.repo_root)) if source_path and source_path.is_relative_to(self.repo_root) else str(source_path or ""),
                    "source_sha256": sha256_file(source_path) if source_path and source_path.exists() else None,
                    "schema_sha256": stable_hash(schema),
                }
            )
        return result

    def start(self, question_id: str, data_path: Path, allowed_names: set[str]) -> dict[str, Any]:
        missing = sorted(allowed_names - set(self.tool_map))
        if missing:
            raise ValueError(f"Tools missing from live claude_tsa registry: {', '.join(missing)}")
        self.session = self.session_manager.create(metadata={"source": "api_sft", "question_id": question_id})
        self.allowed_names = set(allowed_names)
        source = data_path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Question dataset does not exist: {source}")
        upload = self.session.workspace_path / self.dataset_uri
        upload.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, upload)
        return {
            "session_id": self.session.session_id,
            "workspace_path": str(self.session.workspace_path),
            "dataset_uri": self.dataset_uri,
            "source_dataset_sha256": sha256_file(source),
            "staged_dataset_sha256": sha256_file(upload),
            "initial_artifact_count": len(self.artifact_store.list(self.session.session_id)),
        }

    def tools_for_model(self) -> list[dict[str, Any]]:
        if self.session is None:
            raise RuntimeError("Tool runtime session has not started")
        return [
            {
                "type": "function",
                "function": {
                    "name": item["name"],
                    "description": item["description"],
                    "parameters": item["input_schema"],
                },
            }
            for item in self.describe_tools(self.allowed_names, self.session.session_id)
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.session is None:
            raise RuntimeError("Tool runtime session has not started")
        if name not in self.allowed_names:
            raise ValueError(f"Tool is not allowed for this question: {name}")
        tool = self.tool_map[name]
        before = {artifact.artifact_id for artifact in self.artifact_store.list(self.session.session_id)}
        try:
            result = asyncio.run(tool.handler(arguments))
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "result": {
                    "content": [{"type": "text", "text": f"Tool execution failed: {type(exc).__name__}: {exc}"}],
                    "structuredContent": {
                        "ok": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                    "isError": True,
                },
                "created_artifacts": [],
                "image_paths": [],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        json_result = _jsonable(result)
        created = [artifact.to_dict() for artifact in self.artifact_store.list(self.session.session_id) if artifact.artifact_id not in before]
        image_paths = [artifact["path"] for artifact in created if Path(artifact["path"]).suffix.lower() in IMAGE_SUFFIXES]
        if isinstance(json_result, dict):
            raw_structured = json_result.get("structuredContent", {})
            structured = raw_structured if isinstance(raw_structured, dict) else {}
            summary = structured.get("summary")
            summary_error = summary.get("error") if isinstance(summary, dict) else None
            structured_error = structured.get("error")
            ok = (
                not bool(json_result.get("is_error") or json_result.get("isError"))
                and structured.get("ok", True) is not False
                and not summary_error
                and not structured_error
            )
        else:
            ok = True
        execution = {"ok": ok, "result": json_result, "created_artifacts": created, "image_paths": image_paths}
        if not ok:
            execution["error_type"] = "structured_tool_error"
            execution["error"] = str(summary_error or structured_error or "handler returned an error response")
        return execution
