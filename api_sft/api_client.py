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


def _moonshot_schema(value: Any) -> Any:
    """Project JSON Schema into Moonshot's accepted tool-schema dialect."""

    if isinstance(value, list):
        return [_moonshot_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _moonshot_schema(child) for key, child in value.items()}
    choices = result.get("anyOf")
    if isinstance(choices, list) and "type" in result:
        # Moonshot requires a root object type but rejects anyOf beside that
        # type. Keep the descriptive property schemas for generation and rely
        # on the untouched local schema for the authoritative validation.
        result.pop("anyOf")
    return result


def _tools_for_provider(tools: list[dict[str, Any]], base_url: str) -> list[dict[str, Any]]:
    if "api.moonshot.cn" not in base_url.lower():
        return tools
    return _moonshot_schema(tools)


def resolve_api_key(config: dict[str, Any]) -> str:
    env_name=str(config.get("api_key_env") or "").strip()
    if env_name:
        key=os.environ.get(env_name)
        if key:
            return key
    direct=str(config.get("api_key","") or "").strip()
    if direct:
        return direct
    if env_name and (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",env_name) or len(env_name)>80):
        raise RuntimeError("api_key_env appears to contain a secret; use api_key for a YAML secret or set api_key_env to an environment variable name")
    if env_name:
        raise RuntimeError(f"Missing API key: set environment variable '{env_name}', add it to the configured .env file, or set api_key in YAML")
    raise RuntimeError("Missing API key configuration: set api_key_env or api_key")


def image_data_url(path: str | Path) -> str:
    path=Path(path); mime=mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def user_message(text: str, images: list[str] | None = None) -> dict[str, Any]:
    if not images: return {"role":"user","content":text}
    content: list[dict[str,Any]]=[{"type":"text","text":text}]
    content.extend({"type":"image_url","image_url":{"url":image_data_url(path),"detail":"high"}} for path in images)
    return {"role":"user","content":content}


def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned=re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$","",raw.strip(),flags=re.I|re.S)
    try: value=json.loads(cleaned)
    except json.JSONDecodeError:
        start,end=cleaned.find("{"),cleaned.rfind("}")
        if start<0 or end<=start: raise
        value=json.loads(cleaned[start:end+1])
    if not isinstance(value,dict): raise ValueError("Model response must be a JSON object")
    return value


class OpenAICompatibleClient:
    def __init__(self, config: dict[str, Any], default_timeout: float=120, default_retries: int=3):
        self.config=config; self.timeout=float(config.get("timeout_seconds",default_timeout)); self.retries=int(config.get("retries",default_retries))

    def complete_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any],dict[str,Any],float,str|None]:
        key=resolve_api_key(self.config)
        payload={"model":self.config["model"],"messages":messages,"temperature":self.config.get("temperature",.2)}
        if tools:
            payload["tools"]=_tools_for_provider(tools,self.config["base_url"])
            payload["tool_choice"]=tool_choice or self.config.get("tool_choice","auto")
            if "parallel_tool_calls" in self.config:
                payload["parallel_tool_calls"]=bool(self.config["parallel_tool_calls"])
        elif self.config.get("response_format",True):
            payload["response_format"]={"type":"json_object"}
        url=self.config["base_url"].rstrip("/")+"/chat/completions"; last: Exception|None=None
        for attempt in range(self.retries):
            started=time.monotonic()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response=client.post(url,headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json=payload)
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        detail=str(getattr(response,"text","") or "")[:2000].strip()
                        status=getattr(response,"status_code",None) or getattr(exc.response,"status_code","unknown")
                        raise ValueError(f"API returned HTTP {status}: {detail}") from exc
                    body=response.json()
                choice=body["choices"][0]
                message=choice["message"]
                if not isinstance(message,dict): raise ValueError("API choice.message must be an object")
                return message,body.get("usage",{}),time.monotonic()-started,choice.get("finish_reason")
            except (httpx.HTTPError,KeyError,ValueError) as exc:
                last=exc
                if attempt+1<self.retries: time.sleep(2**attempt+random.random())
        raise RuntimeError(f"API request failed after {self.retries} attempts: {last}")

    def complete(self, messages: list[dict[str, Any]]) -> tuple[str,dict[str,Any],float]:
        message,usage,latency,_=self.complete_message(messages)
        content=message.get("content")
        if not isinstance(content,str):
            raise RuntimeError("API response did not contain textual message.content")
        return content,usage,latency
