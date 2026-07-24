from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .api_client import OpenAICompatibleClient, image_data_url, parse_json_object, user_message
from .common import append_jsonl, done_ids, iter_jsonl, stable_hash
from .questions import QUESTION_RUNTIME_FORMAT_VERSION, TOOL_EXECUTION_SYSTEM_PROMPT, TOOL_EXECUTION_SYSTEM_PROMPT_ID
from .tool_runtime import ClaudeTsaToolRuntime


TRAJECTORY_FORMAT_VERSION = "tool_trajectory_v2.1"
COMPETITION_AUDIT_FORMAT_VERSION = "trajectory_competition_audit_v2"
MODEL_CATALOG_SEARCH_TOOL = "model_catalog_search"


class TrajectoryGenerationError(RuntimeError):
    """A terminal candidate error that retains the truthful partial trajectory."""

    def __init__(self, message: str, partial_record: dict[str, Any], error_type: str = "trajectory_generation_error"):
        super().__init__(message)
        self.partial_record = partial_record
        self.error_type = error_type


def _model_catalog_search_definition(session_id: str, scope: str) -> dict[str, Any]:
    task_description = "forecast and foundation inference" if scope == "forecast" else "anomaly detection"
    return {
        "type": "function",
        "function": {
            "name": MODEL_CATALOG_SEARCH_TOOL,
            "description": (
                f"Search the read-only {task_description} model catalog. The catalog scope is fixed by the question; "
                "you must infer all finer data and modeling characteristics from observed evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "const": session_id},
                    "task_labels": {"type": "array", "items": {"type": "string", "minLength": 2}, "maxItems": 6},
                    "runtime_types": {"type": "array", "items": {"type": "string", "minLength": 2}, "maxItems": 6},
                    "packages": {"type": "array", "items": {"type": "string", "minLength": 2}, "maxItems": 6},
                    "name_query": {"type": "string", "minLength": 2},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["session_id"],
                "anyOf": [
                    {"required": ["task_labels"]},
                    {"required": ["runtime_types"]},
                    {"required": ["packages"]},
                    {"required": ["name_query"]},
                ],
                "additionalProperties": False,
            },
        },
    }


def _scope_models(models: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    allowed_tasks = {"forecast", "foundation_inference"} if scope == "forecast" else {"anomaly_detection"}
    return [model for model in models if allowed_tasks & set(model.get("tasks") or [])]


def _search_model_catalog(models: list[dict[str, Any]], scope: str, arguments: dict[str, Any]) -> dict[str, Any]:
    task_labels = {str(value).lower() for value in arguments.get("task_labels", [])}
    runtime_types = {str(value).lower() for value in arguments.get("runtime_types", [])}
    packages = {str(value).lower() for value in arguments.get("packages", [])}
    name_query = str(arguments.get("name_query", "")).strip().lower()
    limit = int(arguments.get("limit", 12))
    candidates: list[dict[str, Any]] = []
    for model in _scope_models(models, scope):
        tasks = {str(value).lower() for value in model.get("tasks", [])}
        runtime = str(model.get("runtime_type", "")).lower()
        package = str(model.get("package", "")).lower()
        name = str(model.get("name", ""))
        if task_labels and not any(label in task or task in label for label in task_labels for task in tasks):
            continue
        if runtime_types and runtime not in runtime_types:
            continue
        if packages and package not in packages:
            continue
        if name_query and name_query not in name.lower():
            continue
        candidates.append({
            "name": name,
            "package": model.get("package"),
            "runtime_type": model.get("runtime_type"),
            "tasks": model.get("tasks", []),
        })
        if len(candidates) >= limit:
            break
    return {
        "content": [{"type": "text", "text": f"Model catalog search returned {len(candidates)} candidates."}],
        "structuredContent": {
            "ok": True,
            "summary": {
                "scope": scope,
                "candidate_count": len(candidates),
                "candidates": candidates,
                "query": {
                    key: arguments[key]
                    for key in ["task_labels", "runtime_types", "packages", "name_query", "limit"]
                    if key in arguments
                },
            },
        },
    }


def _runtime_task(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("task") if isinstance(row.get("task"), dict) else {}


def _runtime_prompt(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("prompt") if isinstance(row.get("prompt"), dict) else {}


def _runtime_resources(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("resources") if isinstance(row.get("resources"), dict) else {}


def _runtime_dataset_path(row: dict[str, Any]) -> str:
    dataset = _runtime_resources(row).get("dataset") or {}
    return str(dataset.get("path") or "")


def _runtime_images(row: dict[str, Any]) -> list[str]:
    return [
        str(item["path"])
        for item in _runtime_resources(row).get("images", [])
        if isinstance(item, dict) and item.get("path")
    ]


def _runtime_scope(row: dict[str, Any]) -> str:
    return str(_runtime_task(row).get("model_catalog_scope", "none"))


def _runtime_user_content(row: dict[str, Any], dataset_uri: str) -> str:
    request = str(_runtime_prompt(row).get("user_request") or "").strip()
    if not request:
        raise ValueError("Runtime question is missing prompt.user_request")
    return request + "\n\n可访问的数据资源：\n" + f"- dataset_path: {dataset_uri}"


def _api_content(text: str, images: list[str]) -> str | list[dict[str, Any]]:
    if not images:
        return text
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for path in images:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(path), "detail": "high"}})
    return content


def neutral_to_api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "user":
            result.append({"role": "user", "content": _api_content(str(message.get("content", "")), list(message.get("images", [])))})
        elif role == "tool":
            result.append({"role": "tool", "tool_call_id": message["tool_call_id"], "content": str(message.get("content", ""))})
            images = list(message.get("images", []))
            if images:
                result.append({
                    "role": "user",
                    "content": _api_content(
                        f"以下图像是工具 {message.get('name')}（调用 {message['tool_call_id']}）刚生成的真实视觉结果，请将其作为该工具结果的一部分。",
                        images,
                    ),
                })
        elif role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
            if message.get("tool_calls"):
                item["tool_calls"] = message["tool_calls"]
            result.append(item)
        else:
            result.append({"role": role, "content": str(message.get("content", ""))})
    return result


def _normalize_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ValueError("assistant.tool_calls must be a list")
    normalized: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            raise ValueError("Each tool call must contain function.name")
        raw_arguments = function.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            arguments = json.loads(raw_arguments)
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            raise ValueError("function.arguments must be a JSON object or encoded object")
        if not isinstance(arguments, dict):
            raise ValueError("function.arguments must decode to an object")
        normalized.append({
            "id": str(call.get("id") or f"call_{index + 1}"),
            "type": "function",
            "function": {"name": str(function["name"]), "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))},
            "parsed_arguments": arguments,
        })
    return normalized


def _sanitize_tool_result(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if key == "path" and isinstance(child, str) and Path(child).is_absolute():
                result[key] = value.get("uri") or f"session_artifact/{Path(child).name}"
            else:
                result[key] = _sanitize_tool_result(child)
        return result
    if isinstance(value, list):
        return [_sanitize_tool_result(item) for item in value]
    return value


def _observed_tool_content(
    execution: dict[str, Any],
    max_chars: int,
    trajectory_budget: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    visible = {
        "ok": bool(execution.get("ok")),
        "tool_response": _sanitize_tool_result(execution["result"]),
        "created_artifacts": [
            {key: artifact.get(key) for key in ["artifact_id", "kind", "uri", "description", "metadata"]}
            for artifact in execution.get("created_artifacts", [])
        ],
    }
    if trajectory_budget is not None:
        visible["trajectory_budget"] = trajectory_budget
    raw = json.dumps(visible, ensure_ascii=False, separators=(",", ":"))
    audit = {"full_result_sha256": stable_hash(visible), "full_result_chars": len(raw), "truncated_for_model": False}
    if len(raw) <= max_chars:
        return raw, audit
    wrapper = {"truncated": True, "full_result_sha256": audit["full_result_sha256"], "full_result_chars": len(raw), "preview": raw[: max(0, max_chars - 240)]}
    audit["truncated_for_model"] = True
    return json.dumps(wrapper, ensure_ascii=False, separators=(",", ":")), audit


def _execution_system_prompt(row: dict[str, Any], setup: dict[str, Any], min_calls: int, max_calls: int) -> str:
    prompt_id = str(_runtime_prompt(row).get("system_prompt_id") or "")
    if prompt_id != TOOL_EXECUTION_SYSTEM_PROMPT_ID:
        raise ValueError(f"Unsupported system_prompt_id: {prompt_id}")
    base = TOOL_EXECUTION_SYSTEM_PROMPT
    scope = _runtime_scope(row)
    catalog_rule = (
        "- 本题不提供模型目录检索；不要编造或无故讨论具体模型名称。\n"
        if scope == "none"
        else f"- 本题可按需检索粗粒度 {scope} 模型目录；数据画像和细粒度模型类型必须由你根据真实结果自主判断。\n"
    )
    return (
        base
        + "\n\n本条轨迹的执行上下文：\n"
        + f"- session_id: {setup['session_id']}\n"
        + f"- 原始数据 URI: {setup['dataset_uri']}\n"
        + "- 初始 session 没有预制 channel；若分析工具需要 channel，应先画像数据，再自行调用 data_convert 创建。\n"
        + f"- 每轮最多调用一个工具；至少完成 {min_calls} 次、最多 {max_calls} 次有信息增益的成功调用。\n"
        + "- 调用参数必须使用上面的 session_id 和数据 URI，不得使用原始绝对路径。\n"
        + "- 工具失败时读取错误并修正参数或换工具；失败调用不计入成功次数，禁止假装成功。\n"
        + "- 不输出隐藏思维链；调用前只写必要的简短决策说明，最终回答使用自然中文。\n"
        + catalog_rule
    )


def _usage_add(total: dict[str, int], usage: dict[str, Any]) -> None:
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        total[key] = total.get(key, 0) + int(usage.get(key, 0) or 0)


def _error_execution(
    error_type: str,
    message: str,
    failure_signature: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error = {"type": error_type, "message": message}
    if details:
        error.update(details)
    return {
        "ok": False,
        "result": {
            "content": [{"type": "text", "text": f"{error_type}: {message}"}],
            "structuredContent": {"ok": False, "error": error},
            "isError": True,
        },
        "created_artifacts": [],
        "image_paths": [],
        "error_type": error_type,
        "error": message,
        "failure_signature": failure_signature or stable_hash({"type": error_type, "message": message}),
    }


def _schema_error_execution(
    name: str,
    schema: dict[str, Any],
    arguments: dict[str, Any],
    errors: list[Any],
) -> dict[str, Any]:
    first = errors[0]
    details = {
        "path": list(first.path),
        "schema_path": list(first.schema_path),
        "validator": first.validator,
        "required_fields": list(schema.get("required", [])),
        "allowed_fields": sorted((schema.get("properties") or {}).keys()),
        "repair_hint": "Use only allowed_fields, include every required field, and preserve the provided session_id const.",
        "validation_errors": [
            {"message": item.message, "path": list(item.path), "validator": item.validator}
            for item in errors[:5]
        ],
    }
    signature = stable_hash({
        "name": name,
        "type": "schema_validation",
        "validator": first.validator,
        "path": list(first.path),
        "arguments": arguments,
    })
    return _error_execution("schema_validation", first.message, signature, details)


def generate_one_trajectory(
    row: dict[str, Any],
    client: OpenAICompatibleClient,
    runtime: ClaudeTsaToolRuntime,
    model_catalog: list[dict[str, Any]] | None = None,
    min_tool_calls: int = 1,
    min_distinct_tools: int = 1,
    max_tool_calls: int = 8,
    max_tool_result_chars: int = 16000,
    candidate_index: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    if row.get("format_version") != QUESTION_RUNTIME_FORMAT_VERSION:
        raise ValueError(f"Unsupported question runtime format: {row.get('format_version')}")
    candidate_names = {str(name) for name in row.get("allowed_tools", []) if str(name)}
    data_path = _runtime_dataset_path(row)
    if not data_path:
        raise ValueError("Runtime question is missing resources.dataset.path")
    setup = runtime.start(row["id"], Path(data_path), candidate_names)
    tools = runtime.tools_for_model()
    scope = _runtime_scope(row)
    if scope not in {"none", "forecast", "anomaly_detection"}:
        raise ValueError(f"Unsupported model_catalog_scope: {scope}")
    if scope != "none":
        tools.append(_model_catalog_search_definition(setup["session_id"], scope))
    tool_definitions = {item["function"]["name"]: item for item in tools}
    question = _runtime_user_content(row, setup["dataset_uri"])
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _execution_system_prompt(row, setup, min_tool_calls, max_tool_calls)},
        {"role": "user", "content": question, "images": _runtime_images(row)},
    ]
    tool_events: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    failure_counts: dict[str, int] = {}
    successful_tools: list[str] = []
    usage_total: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    model_turns: list[dict[str, Any]] = []
    final_answer = ""
    def snapshot(status: str, error_type: str | None = None, error: str | None = None) -> dict[str, Any]:
        try:
            tool_source = runtime.source_snapshot()
        except Exception as exc:  # noqa: BLE001
            tool_source = {"snapshot_error": f"{type(exc).__name__}: {exc}"}
        record = {
            "id": row["id"],
            "status": status,
            "format_version": TRAJECTORY_FORMAT_VERSION,
            "question_record": row,
            "tool_source": tool_source,
            "execution_setup": setup,
            "tools": tools,
            "messages": messages,
            "tool_events": tool_events,
            "successful_tool_calls": len(successful_tools),
            "distinct_successful_tools": sorted(set(successful_tools)),
            "final_answer": final_answer,
            "model": client.config.get("model"),
            "candidate_index": candidate_index,
            "model_turns": model_turns,
            "usage": usage_total,
            "latency_seconds": round(time.monotonic() - started, 3),
        }
        if error_type:
            record["terminal_error_type"] = error_type
        if error:
            record["terminal_error"] = error
        return record

    try:
        for turn_index in range(max_tool_calls + 1):
            budget_exhausted = len(tool_events) >= max_tool_calls
            api_messages = neutral_to_api_messages(messages)
            message, usage, latency, finish_reason = client.complete_message(
                api_messages,
                tools=tools,
                tool_choice="none" if budget_exhausted else "auto",
            )
            _usage_add(usage_total, usage)
            calls = _normalize_tool_calls(message)
            model_turns.append({"turn": turn_index + 1, "latency_seconds": round(latency, 3), "finish_reason": finish_reason, "usage": usage})
            assistant_message: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
            if len(calls) > 1:
                assistant_message["tool_calls"] = [{key: value for key, value in call.items() if key != "parsed_arguments"} for call in calls]
                messages.append(assistant_message)
                raise ValueError("Adaptive trajectories require at most one tool call per assistant turn")
            if not calls:
                final_answer = str(message.get("content") or "").strip()
                if not final_answer:
                    raise ValueError("Final assistant turn has neither a tool call nor textual content")
                if len(successful_tools) < min_tool_calls:
                    raise ValueError(f"Trajectory stopped after {len(successful_tools)} successful tool calls; minimum is {min_tool_calls}")
                if len(set(successful_tools)) < min_distinct_tools:
                    raise ValueError(f"Trajectory used only {len(set(successful_tools))} distinct successful tools; minimum is {min_distinct_tools}")
                messages.append(assistant_message)
                break
            call = calls[0]
            serializable_call = {key: value for key, value in call.items() if key != "parsed_arguments"}
            assistant_message["tool_calls"] = [serializable_call]
            if budget_exhausted:
                messages.append(assistant_message)
                raise TrajectoryGenerationError(
                    "Provider returned a tool call after the trajectory budget was exhausted",
                    snapshot("error", "tool_call_after_budget_exhausted", "Provider ignored tool_choice=none"),
                    "tool_call_after_budget_exhausted",
                )
            call_id = call["id"]
            name = call["function"]["name"]
            arguments = call["parsed_arguments"]
            if call_id in seen_call_ids:
                messages.append(assistant_message)
                raise ValueError(f"Duplicate tool call id: {call_id}")
            seen_call_ids.add(call_id)
            messages.append(assistant_message)
            call_started = time.monotonic()
            provider = "claude_tsa"
            if name not in tool_definitions:
                execution = _error_execution("unknown_tool", f"Tool is not available for this question: {name}", stable_hash({"name": name, "type": "unknown_tool", "arguments": arguments}))
                provider = "api_sft_validation"
            else:
                schema = tool_definitions[name]["function"]["parameters"]
                errors = sorted(Draft202012Validator(schema).iter_errors(arguments), key=lambda item: list(item.path))
                if errors:
                    execution = _schema_error_execution(name, schema, arguments, errors)
                    provider = "api_sft_validation"
                elif name == MODEL_CATALOG_SEARCH_TOOL:
                    execution = {"ok": True, "result": _search_model_catalog(model_catalog or [], scope, arguments), "created_artifacts": [], "image_paths": []}
                    provider = "api_sft_model_catalog"
                else:
                    try:
                        execution = runtime.execute(name, arguments)
                    except Exception as exc:  # noqa: BLE001
                        execution = _error_execution(type(exc).__name__, str(exc), stable_hash({"name": name, "type": type(exc).__name__, "message": str(exc), "arguments": arguments}))
            remaining_calls = max_tool_calls - (len(tool_events) + 1)
            trajectory_budget = None
            if remaining_calls <= 1:
                trajectory_budget = {
                    "max_tool_calls": max_tool_calls,
                    "used_tool_calls": len(tool_events) + 1,
                    "remaining_tool_calls": remaining_calls,
                    "instruction": (
                        "At most one additional tool call remains; use it only if essential, then provide the final answer."
                        if remaining_calls == 1
                        else "No tool calls remain; the next assistant response must be the final answer."
                    ),
                }
            content, result_audit = _observed_tool_content(execution, max_tool_result_chars, trajectory_budget)
            messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": content, "images": list(execution.get("image_paths", []))})
            if execution["ok"]:
                successful_tools.append(name)
            event = {
                "tool_call_id": call_id,
                "name": name,
                "arguments": arguments,
                "provider": provider,
                "ok": bool(execution["ok"]),
                "latency_seconds": round(time.monotonic() - call_started, 3),
                "result": execution["result"],
                "result_audit": result_audit,
                "created_artifacts": execution.get("created_artifacts", []),
                "image_paths": execution.get("image_paths", []),
            }
            if trajectory_budget is not None:
                event["trajectory_budget"] = trajectory_budget
            if not execution["ok"]:
                event["error_type"] = execution.get("error_type", "tool_error")
                event["error"] = execution.get("error", "handler returned an error")
                signature = str(execution.get("failure_signature") or stable_hash({"name": name, "type": event["error_type"], "error": event["error"], "arguments": arguments}))
                event["failure_signature"] = signature
                failure_counts[signature] = failure_counts.get(signature, 0) + 1
            tool_events.append(event)
            if not execution["ok"] and failure_counts[event["failure_signature"]] >= 2:
                raise ValueError(f"Repeated tool failure signature: {name}:{event['error_type']}")
        else:
            raise ValueError("Trajectory loop ended without a final answer")
    except TrajectoryGenerationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TrajectoryGenerationError(str(exc), snapshot("error", type(exc).__name__, str(exc)), type(exc).__name__) from exc
    return snapshot("ok")


def _generate_candidate(
    row: dict[str, Any],
    model_config: dict[str, Any],
    candidate_index: int,
    repo_root: Path,
    workspace_root: Path,
    model_catalog: list[dict[str, Any]],
    max_attempts: int,
    min_tool_calls: int,
    min_distinct_tools: int,
    max_tool_calls: int,
    max_tool_result_chars: int,
) -> dict[str, Any]:
    client = OpenAICompatibleClient(model_config)
    attempt_errors: list[dict[str, Any]] = []
    failed_attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        try:
            runtime = ClaudeTsaToolRuntime(repo_root, workspace_root / f"candidate_{candidate_index + 1}")
            record = generate_one_trajectory(
                row,
                client,
                runtime,
                model_catalog=model_catalog,
                min_tool_calls=min_tool_calls,
                min_distinct_tools=min_distinct_tools,
                max_tool_calls=max_tool_calls,
                max_tool_result_chars=max_tool_result_chars,
                candidate_index=candidate_index,
            )
            record["generation_attempt"] = attempt
            record["prior_attempt_errors"] = attempt_errors
            record["failed_attempts"] = failed_attempts
            return record
        except TrajectoryGenerationError as exc:
            partial = dict(exc.partial_record)
            partial["attempt"] = attempt
            partial["error_type"] = exc.error_type
            partial["error"] = str(exc)
            failed_attempts.append(partial)
            attempt_errors.append({"attempt": attempt, "error_type": exc.error_type, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            attempt_errors.append({"attempt": attempt, "error_type": type(exc).__name__, "error": str(exc)})
            failed_attempts.append({
                "attempt": attempt,
                "status": "error",
                "format_version": TRAJECTORY_FORMAT_VERSION,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "messages": [],
                "tool_events": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
    last_partial = dict(failed_attempts[-1]) if failed_attempts else {}
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for failed in failed_attempts:
        _usage_add(total_usage, failed.get("usage", {}))
    return {
        **last_partial,
        "id": row["id"],
        "status": "error",
        "format_version": TRAJECTORY_FORMAT_VERSION,
        "question_record": row,
        "generation_attempt": max_attempts,
        "attempt_errors": attempt_errors,
        "failed_attempts": failed_attempts,
        "model": model_config.get("model"),
        "candidate_index": candidate_index,
        "usage": total_usage,
        "tool_events": list(last_partial.get("tool_events", [])),
    }


def _candidate_tiebreak_key(candidate: dict[str, Any]) -> tuple[int, int, int, int]:
    events = candidate.get("tool_events", [])
    invalid = sum(not event.get("ok") for event in events)
    tokens = int(candidate.get("usage", {}).get("total_tokens", 0) or 0)
    return invalid, len(events), tokens, int(candidate.get("candidate_index", 0))


def _selector_material(candidate: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    def bounded_result(value: Any, max_chars: int = 6000) -> Any:
        sanitized = _sanitize_tool_result(value)
        raw = json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))
        if len(raw) <= max_chars:
            return sanitized
        return {"truncated": True, "full_result_sha256": stable_hash(sanitized), "full_result_chars": len(raw), "preview": raw[:max_chars]}

    return {
        "candidate_index": candidate.get("candidate_index"),
        "model": candidate.get("model"),
        "final_answer": candidate.get("final_answer"),
        "tool_events": [
            {
                "name": event.get("name"),
                "arguments": event.get("arguments"),
                "ok": event.get("ok"),
                "result": bounded_result(event.get("result")),
            }
            for event in candidate.get("tool_events", [])
        ],
        "hard_review": review,
        "usage": candidate.get("usage", {}),
    }


def _selector_images(row: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    paths: list[str] = []
    manifest: list[dict[str, Any]] = []
    for path in _runtime_images(row):
        if Path(path).is_file() and path not in paths:
            manifest.append({"image_index": len(paths), "source": "initial_question", "filename": Path(path).name})
            paths.append(path)
    for candidate in candidates:
        for event in candidate.get("tool_events", []):
            for path in event.get("image_paths", []):
                if Path(path).is_file() and path not in paths:
                    manifest.append({
                        "image_index": len(paths),
                        "source": "tool_result",
                        "candidate_index": candidate.get("candidate_index"),
                        "tool": event.get("name"),
                        "filename": Path(path).name,
                    })
                    paths.append(path)
    return paths, manifest


def _score_total(item: dict[str, Any]) -> float:
    if isinstance(item.get("total_score"), (int, float)):
        return float(item["total_score"])
    keys = ["evidence_consistency", "tool_efficiency", "business_constraints", "visual_grounding", "failure_recovery"]
    values = [float(item.get(key, 0)) for key in keys]
    return sum(values)


def _select_with_judge(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    selector_config: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    images, image_manifest = _selector_images(row, candidates)
    payload = {
        "task": "Choose the better real tool-execution trajectory. Do not add facts or use hidden rubric fields.",
        "question": _runtime_prompt(row).get("user_request"),
        "input_mode": _runtime_task(row).get("input_mode"),
        "image_manifest": image_manifest,
        "candidates": [_selector_material(candidate, review) for candidate, review in zip(candidates, reviews)],
        "scoring": {
            "range": "0-5 per dimension",
            "dimensions": ["evidence_consistency", "tool_efficiency", "business_constraints", "visual_grounding", "failure_recovery"],
            "required_output": {
                "candidate_scores": [{"candidate_index": 0, "evidence_consistency": 0, "tool_efficiency": 0, "business_constraints": 0, "visual_grounding": 0, "failure_recovery": 0, "total_score": 0, "reason": ""}],
                "winner": 0,
                "rationale": "",
            },
        },
    }
    client = OpenAICompatibleClient(selector_config)
    started = time.monotonic()
    raw, usage, latency = client.complete([
        {"role": "system", "content": "You are a strict multimodal judge for time-series tool trajectories. Output one valid JSON object only."},
        user_message(json.dumps(payload, ensure_ascii=False), images),
    ])
    parsed = parse_json_object(raw)
    score_rows = parsed.get("candidate_scores")
    if not isinstance(score_rows, list):
        raise ValueError("Selector response missing candidate_scores")
    scores: dict[int, float] = {}
    for item in score_rows:
        if isinstance(item, dict) and int(item.get("candidate_index", -1)) in {0, 1}:
            scores[int(item["candidate_index"])] = _score_total(item)
    if set(scores) != {0, 1}:
        raise ValueError("Selector must score both candidate indexes")
    if scores[0] == scores[1]:
        winner = min(candidates, key=_candidate_tiebreak_key)["candidate_index"]
        decision_rule = "deterministic_tiebreak"
    else:
        winner = max(scores, key=scores.get)
        decision_rule = "judge_score"
    return int(winner), {
        "status": "ok",
        "model": selector_config.get("model"),
        "decision_rule": decision_rule,
        "response": parsed,
        "usage": usage,
        "latency_seconds": round(latency, 3),
        "wall_seconds": round(time.monotonic() - started, 3),
        "image_manifest": image_manifest,
    }


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
        raise RuntimeError(f"Trajectories require {QUESTION_RUNTIME_FORMAT_VERSION}; regenerate stale questions first: {', '.join(map(str, stale[:3]))}")
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
        from .trajectory_verify import deterministic_trajectory_review

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
                selector_audit = {"status": "error", "model": selector_config.get("model"), "error_type": type(exc).__name__, "error": str(exc)}
        else:
            selection_rule = "both_hard_gate_failed"
        audit = {
            "id": row["id"],
            "format_version": COMPETITION_AUDIT_FORMAT_VERSION,
            "candidate_count": 2,
            "candidates": candidates,
            "hard_reviews": reviews,
            "selection_rule": selection_rule,
            "winner_index": winner_index,
            "selector": selector_audit,
        }
        append_jsonl(audit_path, audit)
        if winner_index is None:
            output = {
                "id": row["id"],
                "status": "error",
                "format_version": TRAJECTORY_FORMAT_VERSION,
                "question_record": row,
                "error": "both_trajectory_candidates_failed_hard_validation",
                "competition": {"selection_rule": selection_rule, "audit_id": row["id"]},
            }
        else:
            output = dict(candidates[winner_index])
            output["competition"] = {
                "selection_rule": selection_rule,
                "winner_index": winner_index,
                "candidate_generators": [candidate.get("model") for candidate in candidates],
                "hard_review_scores": [review.get("score") for review in reviews],
                "selector_model": selector_config.get("model") if selector_audit else None,
                "audit_id": row["id"],
            }
        append_jsonl(output_path, output)
