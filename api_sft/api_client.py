from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx


RETRYABLE_HTTP_STATUSES = {408, 409, 429}


class APIRequestError(RuntimeError):
    """A redacted API failure with enough structure for retry/circuit-breaker logic."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None,
        retryable: bool,
        attempts: int,
        detail: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.attempts = attempts
        self.detail = detail

    def audit_summary(self) -> dict[str, Any]:
        return {
            "error_type": type(self).__name__,
            "status_code": self.status_code,
            "retryable": self.retryable,
            "attempts": self.attempts,
            "detail": self.detail,
        }


def _moonshot_schema(value: Any) -> Any:
    """Retain the original pipeline's official-Moonshot compatibility behavior."""

    if isinstance(value, list):
        return [_moonshot_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _moonshot_schema(child) for key, child in value.items()}
    choices = result.get("anyOf")
    if isinstance(choices, list) and "type" in result:
        result.pop("anyOf")
    return result


def _tools_for_provider(
    tools: list[dict[str, Any]],
    base_url: str,
    *,
    preprojected: bool = False,
) -> list[dict[str, Any]]:
    if preprojected or "api.moonshot.cn" not in base_url.lower():
        return tools
    return _moonshot_schema(tools)


def resolve_api_key(config: dict[str, Any]) -> str:
    """Resolve shared-pipeline credentials while never echoing a secret-like value."""

    env_name = str(config.get("api_key_env") or "").strip()
    if env_name:
        key = os.environ.get(env_name)
        if key:
            return key
    direct = str(config.get("api_key", "") or "").strip()
    if direct:
        return direct
    if env_name and (
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name) or len(env_name) > 80
    ):
        raise RuntimeError(
            "api_key_env appears to contain a secret; use api_key for a YAML secret "
            "or set api_key_env to an environment variable name"
        )
    if env_name:
        raise RuntimeError(
            f"Missing API key: set environment variable '{env_name}', add it to the "
            "configured .env file, or set api_key in YAML"
        )
    raise RuntimeError("Missing API key configuration: set api_key_env or api_key")


def image_data_url(path: str | Path) -> str:
    image_path = Path(path)
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def user_message(text: str, images: list[str] | None = None) -> dict[str, Any]:
    if not images:
        return {"role": "user", "content": text}
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": image_data_url(path), "detail": "high"},
        }
        for path in images
    )
    return {"role": "user", "content": content}


def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        raw.strip(),
        flags=re.I | re.S,
    )
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


def chat_completions_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("base_url must be a non-empty string")
    if normalized.lower().endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def _redact(value: Any, secret: str) -> str:
    text = str(value or "").strip()[:2000]
    return text.replace(secret, "<redacted>") if secret else text


def _retryable_status(status_code: int) -> bool:
    return status_code in RETRYABLE_HTTP_STATUSES or status_code >= 500


class OpenAICompatibleClient:
    def __init__(
        self,
        config: dict[str, Any],
        default_timeout: float = 120,
        default_retries: int = 3,
    ):
        self.config = config
        self.timeout = float(config.get("timeout_seconds", default_timeout))
        self.retries = int(config.get("retries", default_retries))
        if self.retries < 1:
            raise ValueError("retries must be at least 1")
        self._reuse_http_client = bool(config.get("_reuse_http_client", False))
        self._http_client: httpx.Client | None = None

    def _client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=self.timeout)
        return self._http_client

    def close(self) -> None:
        if self._http_client is None:
            return
        close = getattr(self._http_client, "close", None)
        if callable(close):
            close()
        self._http_client = None

    def _post(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Any:
        if self._reuse_http_client:
            return self._client().post(url, headers=headers, json=payload)
        with httpx.Client(timeout=self.timeout) as client:
            return client.post(url, headers=headers, json=payload)

    def complete_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], float, str | None]:
        key = resolve_api_key(self.config)
        payload: dict[str, Any] = {
            "model": self.config["model"],
            "messages": messages,
        }
        if "temperature" in self.config:
            payload["temperature"] = self.config["temperature"]
        elif not self.config.get("_omit_default_temperature", False):
            payload["temperature"] = 0.2
        if tools:
            payload["tools"] = _tools_for_provider(
                tools,
                self.config["base_url"],
                preprojected=bool(self.config.get("_tools_preprojected", False)),
            )
            payload["tool_choice"] = tool_choice or self.config.get("tool_choice", "auto")
            if "parallel_tool_calls" in self.config:
                payload["parallel_tool_calls"] = bool(self.config["parallel_tool_calls"])
        elif self.config.get("response_format", True):
            payload["response_format"] = {"type": "json_object"}

        url = chat_completions_url(self.config["base_url"])
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        last_error: APIRequestError | None = None
        for attempt in range(1, self.retries + 1):
            started = time.monotonic()
            try:
                response = self._post(url, headers, payload)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    status = int(
                        getattr(response, "status_code", None)
                        or getattr(exc.response, "status_code", 0)
                        or 0
                    )
                    detail = _redact(getattr(response, "text", ""), key)
                    retryable = _retryable_status(status)
                    raise APIRequestError(
                        f"API returned HTTP {status}",
                        status_code=status,
                        retryable=retryable,
                        attempts=attempt,
                        detail=detail or None,
                    ) from exc
                try:
                    body = response.json()
                    choice = body["choices"][0]
                    message = choice["message"]
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise APIRequestError(
                        "API response is not a valid OpenAI-compatible chat completion",
                        status_code=getattr(response, "status_code", None),
                        retryable=False,
                        attempts=attempt,
                        detail=_redact(type(exc).__name__, key),
                    ) from exc
                if not isinstance(message, dict):
                    raise APIRequestError(
                        "API choice.message must be an object",
                        status_code=getattr(response, "status_code", None),
                        retryable=False,
                        attempts=attempt,
                    )
                return (
                    message,
                    body.get("usage", {}),
                    time.monotonic() - started,
                    choice.get("finish_reason"),
                )
            except APIRequestError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self.retries:
                    raise
            except httpx.HTTPError as exc:
                last_error = APIRequestError(
                    f"API transport failed: {type(exc).__name__}",
                    status_code=None,
                    retryable=True,
                    attempts=attempt,
                    detail=_redact(str(exc), key) or None,
                )
                if attempt >= self.retries:
                    raise last_error from exc
            time.sleep(2 ** (attempt - 1) + random.random())

        assert last_error is not None
        raise last_error

    def complete(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any], float]:
        message, usage, latency, _ = self.complete_message(messages)
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("API response did not contain textual message.content")
        return content, usage, latency
