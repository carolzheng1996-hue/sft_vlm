from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from api_sft.common import stable_hash
from api_sft.trajectories import _sanitize_tool_result


_UNSUPPORTED_SCHEMA_KEYS = {
    "$defs",
    "$id",
    "$ref",
    "$schema",
    "additionalProperties",
    "allOf",
    "anyOf",
    "default",
    "definitions",
    "dependentSchemas",
    "else",
    "examples",
    "if",
    "not",
    "oneOf",
    "patternProperties",
    "then",
    "unevaluatedProperties",
}


class ToolSchemaCompatibilityError(RuntimeError):
    """A request schema cannot be safely represented by the portable API subset."""


@dataclass
class SessionBinding:
    session_id: str | None = None


def _without_session_parameter(schema: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(schema)
    properties = result.get("properties")
    if isinstance(properties, dict):
        properties.pop("session_id", None)
    required = result.get("required")
    if isinstance(required, list):
        required = [name for name in required if name != "session_id"]
        if required:
            result["required"] = required
        else:
            result.pop("required", None)
    return result


def model_tools_from_execution(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Return the authoritative model-facing tools without runtime-only fields."""

    result = copy.deepcopy(tools or [])
    for tool in result:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        parameters = function.get("parameters")
        if isinstance(parameters, dict):
            function["parameters"] = _without_session_parameter(parameters)
    return result


def _project_schema(value: Any, path: str = "$") -> Any:
    if isinstance(value, list):
        return [_project_schema(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if not isinstance(value, dict):
        return value

    result: dict[str, Any] = {}
    for key, child in value.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS or key in {"const", "enum", "type"}:
            continue
        result[key] = _project_schema(child, f"{path}.{key}")

    raw_type = value.get("type")
    removed_null = False
    if isinstance(raw_type, list):
        concrete = [item for item in raw_type if item != "null"]
        removed_null = len(concrete) != len(raw_type)
        if not concrete:
            raise ToolSchemaCompatibilityError(
                f"Portable tool schema has no usable non-null type at {path}"
            )
        if len(concrete) == 1:
            result["type"] = concrete[0]
        # Multiple concrete primitive types are safely relaxed by omitting
        # `type`; the untouched execution schema remains authoritative.
    elif raw_type is not None:
        result["type"] = _project_schema(raw_type, f"{path}.type")

    if "const" in value:
        enum_values = [_project_schema(value["const"], f"{path}.const")]
    else:
        raw_enum = value.get("enum")
        enum_values = (
            [_project_schema(item, f"{path}.enum") for item in raw_enum]
            if isinstance(raw_enum, list)
            else None
        )
    if enum_values is not None:
        if removed_null:
            enum_values = [item for item in enum_values if item is not None]
        if not enum_values:
            raise ToolSchemaCompatibilityError(
                f"Portable tool schema produced an empty enum at {path}"
            )
        result["enum"] = enum_values
    return result


def _validate_enum_types(value: Any, path: str) -> None:
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_enum_types(child, f"{path}[{index}]")
        return
    if not isinstance(value, dict):
        return
    schema_type = value.get("type")
    enum_values = value.get("enum")
    if isinstance(schema_type, str) and isinstance(enum_values, list):
        validator = Draft202012Validator({"type": schema_type})
        invalid = [item for item in enum_values if not validator.is_valid(item)]
        if invalid:
            raise ToolSchemaCompatibilityError(
                f"Enum values do not match type {schema_type!r} at {path}: {invalid!r}"
            )
    for key, child in value.items():
        _validate_enum_types(child, f"{path}.{key}")


def project_tools_for_api(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Create a portable request-schema copy without mutating authoritative tools."""

    copied = copy.deepcopy(tools or [])
    for index, tool in enumerate(copied):
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            raise ToolSchemaCompatibilityError(
                f"Tool definition at index {index} is missing function metadata"
            )
        name = str(function.get("name") or f"index_{index}")
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            raise ToolSchemaCompatibilityError(
                f"Tool {name} is missing an object parameters schema"
            )
        function["parameters"] = _project_schema(parameters, f"tool.{name}.parameters")
        _validate_enum_types(function["parameters"], f"tool.{name}.parameters")
        try:
            Draft202012Validator.check_schema(function["parameters"])
        except Exception as exc:  # noqa: BLE001 - report the exact tool boundary
            raise ToolSchemaCompatibilityError(
                f"Projected schema for tool {name} is invalid: {exc}"
            ) from exc
    return copied


def _schema_types(schema: dict[str, Any]) -> set[str]:
    value = schema.get("type")
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def _decode_json_containers(value: Any, schema: dict[str, Any]) -> tuple[Any, int]:
    """Decode only JSON-encoded arrays/objects that validate losslessly."""

    decoded = 0
    expected = _schema_types(schema)
    if isinstance(value, str) and expected & {"array", "object"}:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        wanted_type = (
            ("array" in expected and isinstance(parsed, list))
            or ("object" in expected and isinstance(parsed, dict))
        )
        if wanted_type and not list(Draft202012Validator(schema).iter_errors(parsed)):
            value = parsed
            decoded += 1

    if isinstance(value, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, child in list(value.items()):
                child_schema = properties.get(key)
                if isinstance(child_schema, dict):
                    value[key], count = _decode_json_containers(child, child_schema)
                    decoded += count
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        item_schema = schema["items"]
        for index, child in enumerate(value):
            value[index], count = _decode_json_containers(child, item_schema)
            decoded += count
    return value, decoded


def _clean_runtime_value(value: Any, session_id: str | None = None) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if key == "session_id":
                continue
            if key in {"required_fields", "allowed_fields"} and isinstance(child, list):
                child = [item for item in child if item != "session_id"]
            result[key] = _clean_runtime_value(child, session_id)
        return result
    if isinstance(value, list):
        return [_clean_runtime_value(item, session_id) for item in value]
    if isinstance(value, str):
        result = value
        if session_id:
            result = result.replace(session_id, "runtime_session")
        result = result.replace(
            "preserve the provided session_id const",
            "correct the reported business fields",
        )
        return result
    return value


def _clean_system_prompt(content: str, session_id: str | None = None) -> str:
    lines: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("- session_id:"):
            continue
        if "调用参数必须使用上面的 session_id 和数据 URI" in line:
            line = "- 调用参数必须使用上面的数据 URI，不得使用原始绝对路径；会话参数由 runtime 自动注入。"
        if session_id:
            line = line.replace(session_id, "runtime_session")
        lines.append(line)
    return "\n".join(lines)


def _model_arguments(raw_arguments: Any) -> tuple[dict[str, Any] | None, bool]:
    if isinstance(raw_arguments, dict):
        return copy.deepcopy(raw_arguments), False
    if not isinstance(raw_arguments, str):
        return None, False
    try:
        value = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return None, False
    return (value, True) if isinstance(value, dict) else (None, False)


def _clean_tool_call(call: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    result = copy.deepcopy(call)
    function = result.get("function")
    if not isinstance(function, dict):
        return result
    arguments, _ = _model_arguments(function.get("arguments", "{}"))
    if arguments is None:
        return result
    arguments.pop("session_id", None)
    function["arguments"] = json.dumps(
        arguments,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return result


def _clean_tool_content(content: str, session_id: str | None = None) -> str:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return str(_clean_runtime_value(content, session_id))
    return json.dumps(
        _clean_runtime_value(value, session_id),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def model_messages_from_execution(
    messages: list[dict[str, Any]],
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    result = copy.deepcopy(messages)
    for message in result:
        role = message.get("role")
        if role == "system" and isinstance(message.get("content"), str):
            message["content"] = _clean_system_prompt(message["content"], session_id)
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            message["tool_calls"] = [
                _clean_tool_call(call, session_id)
                for call in message["tool_calls"]
                if isinstance(call, dict)
            ]
        if role == "tool" and isinstance(message.get("content"), str):
            message["content"] = _clean_tool_content(message["content"], session_id)
    return result


class SessionBoundRuntime:
    """Bind runtime infrastructure to the trajectory without exposing it to the model."""

    def __init__(self, delegate: Any, binding: SessionBinding):
        self.delegate = delegate
        self.binding = binding

    def start(self, question_id: str, data_path: Any, allowed_names: set[str]) -> dict[str, Any]:
        setup = self.delegate.start(question_id, data_path, allowed_names)
        session_id = str(setup.get("session_id") or "").strip()
        if not session_id:
            raise RuntimeError("Tool runtime did not provide a session_id")
        self.binding.session_id = session_id
        return setup

    def tools_for_model(self) -> list[dict[str, Any]]:
        # The shared generator needs the execution schema for its second validation pass.
        return self.delegate.tools_for_model()

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        session_id = self.binding.session_id
        if not session_id or arguments.get("session_id") != session_id:
            raise RuntimeError("Runtime session binding was not applied")
        result = self.delegate.execute(name, arguments)
        return _clean_runtime_value(result, session_id)

    def source_snapshot(self) -> dict[str, Any]:
        return self.delegate.source_snapshot()


class ProviderCompatibleClient:
    """Model-independent boundary adapter for schemas, arguments, and calls."""

    def __init__(self, delegate: Any, binding: SessionBinding):
        self.delegate = delegate
        self.binding = binding
        self.config = delegate.config
        self.schema_projection_applied = False
        self.argument_json_decoded = 0
        self.multi_call_retried = 0
        self.session_argument_ignored = 0
        self.session_injected = False
        self.adapter_error: str | None = None

    @staticmethod
    def _usage_add(total: dict[str, int], usage: dict[str, Any]) -> None:
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            total[key] = total.get(key, 0) + int(usage.get(key, 0) or 0)

    @staticmethod
    def _tool_call_count(message: dict[str, Any]) -> int:
        calls = message.get("tool_calls") or []
        return len(calls) if isinstance(calls, list) else 0

    def _prepare_response(
        self,
        message: dict[str, Any],
        model_tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        result = copy.deepcopy(message)
        definitions = {
            str(tool.get("function", {}).get("name")): tool
            for tool in model_tools
            if tool.get("function", {}).get("name")
        }
        calls = result.get("tool_calls") or []
        if not isinstance(calls, list):
            return result
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "")
            arguments, _ = _model_arguments(function.get("arguments", "{}"))
            if arguments is None:
                continue
            if "session_id" in arguments:
                self.session_argument_ignored += 1
                arguments.pop("session_id", None)
            definition = definitions.get(name)
            if definition is not None:
                schema = definition["function"]["parameters"]
                arguments, count = _decode_json_containers(arguments, schema)
                self.argument_json_decoded += count
            if definition is not None:
                session_id = self.binding.session_id
                if not session_id:
                    raise RuntimeError("Runtime session is not bound before tool generation")
                arguments["session_id"] = session_id
                self.session_injected = True
            function["arguments"] = json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return result

    def complete_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], float, str | None]:
        session_id = self.binding.session_id
        model_messages = model_messages_from_execution(messages, session_id)
        model_tools = model_tools_from_execution(tools)
        request_tools = project_tools_for_api(model_tools)
        self.schema_projection_applied = self.schema_projection_applied or request_tools != model_tools

        message, usage, latency, finish_reason = self.delegate.complete_message(
            model_messages,
            tools=request_tools or None,
            tool_choice=tool_choice,
        )
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._usage_add(total_usage, usage)
        total_latency = float(latency)

        if self._tool_call_count(message) > 1 and tool_choice != "none":
            self.multi_call_retried += 1
            correction = {
                "role": "user",
                "content": (
                    "上一响应包含多个工具调用。当前 adaptive trajectory 每轮只能调用一个工具；"
                    "请只返回此刻最必要的一个工具调用，不要并行调用，也不要执行其余调用。"
                ),
            }
            message, retry_usage, retry_latency, finish_reason = self.delegate.complete_message(
                [*model_messages, correction],
                tools=request_tools or None,
                tool_choice=tool_choice,
            )
            self._usage_add(total_usage, retry_usage)
            total_latency += float(retry_latency)
            if self._tool_call_count(message) > 1:
                self.adapter_error = "provider_returned_multiple_tool_calls_after_retry"

        return (
            self._prepare_response(message, model_tools),
            total_usage,
            total_latency,
            finish_reason,
        )

    def audit_summary(self) -> dict[str, Any]:
        return {
            "protocol": "openai_compatible",
            "session_injected": self.session_injected,
            "session_argument_ignored": self.session_argument_ignored,
            "schema_projection_applied": self.schema_projection_applied,
            "argument_json_decoded": self.argument_json_decoded,
            "multi_call_retried": self.multi_call_retried,
            "adapter_error": self.adapter_error,
        }


def _event_visible(event: dict[str, Any]) -> dict[str, Any]:
    visible = {
        "ok": bool(event.get("ok")),
        "tool_response": _sanitize_tool_result(event.get("result")),
        "created_artifacts": [
            {
                key: artifact.get(key)
                for key in ["artifact_id", "kind", "uri", "description", "metadata"]
            }
            for artifact in event.get("created_artifacts", [])
            if isinstance(artifact, dict)
        ],
    }
    if event.get("trajectory_budget") is not None:
        visible["trajectory_budget"] = event["trajectory_budget"]
    return visible


def sanitize_candidate_record(
    record: dict[str, Any],
    binding: SessionBinding,
    adapter_summary: dict[str, Any],
) -> dict[str, Any]:
    """Project an execution record back to the exact model/training protocol."""

    result = copy.deepcopy(record)
    session_id = binding.session_id
    result["tools"] = model_tools_from_execution(result.get("tools") or [])
    result["messages"] = model_messages_from_execution(result.get("messages") or [], session_id)
    for event in result.get("tool_events") or []:
        arguments = event.get("arguments")
        if isinstance(arguments, dict):
            arguments.pop("session_id", None)
        event["result"] = _clean_runtime_value(event.get("result"), session_id)
        event["created_artifacts"] = _clean_runtime_value(
            event.get("created_artifacts") or [],
            session_id,
        )
        if isinstance(event.get("error"), str):
            event["error"] = _clean_runtime_value(event["error"], session_id)
        visible = _event_visible(event)
        raw = json.dumps(visible, ensure_ascii=False, separators=(",", ":"))
        result_audit = event.setdefault("result_audit", {})
        result_audit["full_result_sha256"] = stable_hash(visible)
        result_audit["full_result_chars"] = len(raw)

    setup = result.get("execution_setup")
    if isinstance(setup, dict):
        result["execution_setup"] = {
            key: copy.deepcopy(setup[key])
            for key in [
                "dataset_uri",
                "source_dataset_sha256",
                "staged_dataset_sha256",
                "initial_artifact_count",
            ]
            if key in setup
        }
        result["execution_setup"]["session_injected"] = True
    result["provider_adapter"] = copy.deepcopy(adapter_summary)
    return result
