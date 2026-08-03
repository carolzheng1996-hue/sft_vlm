from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from api_sft.api_client import (
    APIRequestError,
    OpenAICompatibleClient,
    chat_completions_url,
)


class _Response:
    def __init__(self, status_code: int, body=None, text: str = ""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            response = httpx.Response(self.status_code)
            raise httpx.HTTPStatusError(
                str(self.status_code),
                request=httpx.Request("POST", "https://platform.test/v1/chat/completions"),
                response=response,
            )

    def json(self):
        return self._body


class ApiCompatibilityTests(unittest.TestCase):
    def test_chat_completions_url_accepts_root_or_full_endpoint(self):
        self.assertEqual(
            chat_completions_url("https://platform.test/v1"),
            "https://platform.test/v1/chat/completions",
        )
        self.assertEqual(
            chat_completions_url("https://platform.test/v1/chat/completions/"),
            "https://platform.test/v1/chat/completions",
        )

    def test_single_candidate_payload_omits_unconfigured_temperature(self):
        requests = []

        class FakeClient:
            def __init__(self, **kwargs):
                self.closed = False

            def post(self, url, headers, json):
                requests.append((url, headers, json))
                return _Response(
                    200,
                    {"choices": [{"message": {"content": "ok"}}], "usage": {}},
                )

            def close(self):
                self.closed = True

        config = {
            "base_url": "https://platform.test/v1",
            "model": "claude-sonnet",
            "api_key": "test-secret",
            "_omit_default_temperature": True,
            "_reuse_http_client": True,
        }
        with patch("api_sft.api_client.httpx.Client", FakeClient):
            client = OpenAICompatibleClient(config)
            client.complete_message([{"role": "user", "content": "hello"}])
            client.close()

        self.assertEqual(requests[0][0], "https://platform.test/v1/chat/completions")
        self.assertEqual(requests[0][2]["model"], "claude-sonnet")
        self.assertNotIn("temperature", requests[0][2])

    def test_non_retryable_http_error_fails_once_and_redacts_api_key(self):
        calls = 0
        secret = "secret-that-must-not-leak"

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def post(self, *args, **kwargs):
                nonlocal calls
                calls += 1
                return _Response(400, text=f"invalid schema {secret}")

            def close(self):
                pass

        config = {
            "base_url": "https://platform.test/v1",
            "model": "kimi-k3",
            "api_key": secret,
            "retries": 3,
            "_reuse_http_client": True,
        }
        with patch("api_sft.api_client.httpx.Client", FakeClient):
            client = OpenAICompatibleClient(config)
            with self.assertRaises(APIRequestError) as caught:
                client.complete_message([{"role": "user", "content": "hello"}])
            client.close()

        self.assertEqual(calls, 1)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertFalse(caught.exception.retryable)
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(secret, str(caught.exception.audit_summary()))

    def test_retryable_http_error_retries_then_succeeds(self):
        calls = 0

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def post(self, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return _Response(429, text="rate limited")
                return _Response(
                    200,
                    {"choices": [{"message": {"content": "ok"}}], "usage": {}},
                )

            def close(self):
                pass

        config = {
            "base_url": "https://platform.test/v1/chat/completions",
            "model": "glm",
            "api_key": "test-secret",
            "retries": 2,
            "_reuse_http_client": True,
        }
        with (
            patch("api_sft.api_client.httpx.Client", FakeClient),
            patch("api_sft.api_client.time.sleep"),
        ):
            client = OpenAICompatibleClient(config)
            message, _, _, _ = client.complete_message(
                [{"role": "user", "content": "hello"}]
            )
            client.close()

        self.assertEqual(calls, 2)
        self.assertEqual(message["content"], "ok")


if __name__ == "__main__":
    unittest.main()
