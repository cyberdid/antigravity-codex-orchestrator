"""Ollama HTTP streaming client for on-device Gemma 4 12B model."""

import json
import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

import urllib.error
import urllib.request

from antigravity_hybrid_orchestrator.workforce.context import chat_num_predict, num_ctx, ollama_timeout_s


class OllamaConnectionError(Exception):
    """Raised when connection to local Ollama service fails."""
    pass


class WorkforceClient:
    """Client for streaming token generation from local Ollama service."""

    DEFAULT_BASE_URL = "http://127.0.0.1:11434"
    DEFAULT_MODEL = "gemma4:12b"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = (model or os.environ.get("OLLAMA_MODEL", "")).strip() or self.DEFAULT_MODEL
        self.timeout = timeout if timeout is not None else ollama_timeout_s()


    def check_health(self) -> bool:
        """Check if local Ollama daemon is reachable."""
        url = f"{self.base_url}/api/version"
        try:
            if HAS_HTTPX:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(url)
                    return resp.status_code == 200
            else:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    return resp.status == 200
        except Exception:
            return False

    def warmup(self) -> bool:
        """Pre-warm or verify the model is loaded in Ollama to prevent cold-start latency."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": "ping",
            "stream": False,
            "options": {"num_predict": 1},
        }
        try:
            if HAS_HTTPX:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(url, json=payload)
                    return resp.status_code == 200
            else:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60.0) as resp:
                    return resp.status == 200
        except Exception:
            return False

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Generator[Tuple[str, str, Dict[str, Any]], None, None]:
        """Stream token generation from Ollama HTTP API.

        Yields:
            Tuple of (token_chunk, accumulated_text, metadata_dict).
            The final yield contains done=True and evaluation metrics.

        Raises:
            OllamaConnectionError: If connection fails or stream is interrupted.
        """
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2048,
                "num_ctx": num_ctx(),
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        accumulated_text = ""

        if HAS_HTTPX:
            # Use robust httpx streaming client with separate connect/read timeouts
            client_timeout = httpx.Timeout(
                timeout=self.timeout,
                connect=60.0,
                read=self.timeout,
                write=60.0,
                pool=60.0,
            )
            try:
                with httpx.Client(timeout=client_timeout) as client:
                    with client.stream("POST", url, json=payload) as resp:
                        if resp.status_code != 200:
                            raise OllamaConnectionError(
                                f"Ollama HTTP error {resp.status_code}: {resp.read().decode('utf-8')}"
                            )

                        for line in resp.iter_lines():
                            line = line.strip()
                            if not line:
                                continue

                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            token = chunk.get("response", "")
                            accumulated_text += token

                            is_done = chunk.get("done", False)
                            meta: Dict[str, Any] = {"done": is_done}

                            if is_done:
                                eval_count = int(chunk.get("eval_count", 0))
                                eval_duration_ns = int(chunk.get("eval_duration", 0))
                                eval_duration_s = (
                                    eval_duration_ns / 1_000_000_000.0 if eval_duration_ns > 0 else 0.0
                                )
                                prompt_eval_count = int(chunk.get("prompt_eval_count", 0))
                                total_duration_ns = int(chunk.get("total_duration", 0))
                                total_duration_s = (
                                    total_duration_ns / 1_000_000_000.0 if total_duration_ns > 0 else 0.0
                                )

                                meta.update({
                                    "eval_count": eval_count,
                                    "eval_duration_s": eval_duration_s,
                                    "prompt_eval_count": prompt_eval_count,
                                    "total_duration_s": total_duration_s,
                                })

                            yield token, accumulated_text, meta

            except httpx.ConnectError as e:
                raise OllamaConnectionError(f"Failed to connect to Ollama at {url}: {e}") from e
            except httpx.TimeoutException as e:
                raise OllamaConnectionError(f"Ollama streaming generation timed out after {self.timeout}s: {e}") from e
            except Exception as e:
                raise OllamaConnectionError(f"Ollama streaming generation error: {e}") from e

        else:
            # Fallback to urllib
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8").strip()
                        if not line:
                            continue

                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        token = chunk.get("response", "")
                        accumulated_text += token

                        is_done = chunk.get("done", False)
                        meta: Dict[str, Any] = {"done": is_done}

                        if is_done:
                            eval_count = int(chunk.get("eval_count", 0))
                            eval_duration_ns = int(chunk.get("eval_duration", 0))
                            eval_duration_s = (
                                eval_duration_ns / 1_000_000_000.0 if eval_duration_ns > 0 else 0.0
                            )
                            prompt_eval_count = int(chunk.get("prompt_eval_count", 0))
                            total_duration_ns = int(chunk.get("total_duration", 0))
                            total_duration_s = (
                                total_duration_ns / 1_000_000_000.0 if total_duration_ns > 0 else 0.0
                            )

                            meta.update({
                                "eval_count": eval_count,
                                "eval_duration_s": eval_duration_s,
                                "prompt_eval_count": prompt_eval_count,
                                "total_duration_s": total_duration_s,
                            })

                        yield token, accumulated_text, meta

            except urllib.error.URLError as e:
                raise OllamaConnectionError(f"Failed to connect to Ollama at {url}: {e}") from e
            except Exception as e:
                raise OllamaConnectionError(f"Ollama streaming generation error: {e}") from e

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST a JSON payload and return the parsed JSON response (non-streaming)."""
        if HAS_HTTPX:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code != 200:
                        raise OllamaConnectionError(
                            f"Ollama HTTP error {resp.status_code}: {resp.text}"
                        )
                    return resp.json()
            except httpx.ConnectError as e:
                raise OllamaConnectionError(f"Failed to connect to Ollama at {url}: {e}") from e
            except httpx.TimeoutException as e:
                raise OllamaConnectionError(f"Ollama request timed out after {self.timeout}s: {e}") from e
            except OllamaConnectionError:
                raise
            except Exception as e:
                raise OllamaConnectionError(f"Ollama request error: {e}") from e
        else:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as e:
                raise OllamaConnectionError(f"Failed to connect to Ollama at {url}: {e}") from e
            except Exception as e:
                raise OllamaConnectionError(f"Ollama request error: {e}") from e

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """Non-streaming `/api/chat` call with native Ollama tool-calling support.

        Args:
            messages: Chat history (roles: system/user/assistant/tool).
            tools: Optional tool schemas (OpenAI-style function definitions).
            temperature: Sampling temperature.

        Returns:
            {"content": str, "tool_calls": [{"name", "arguments"}], "input_tokens", "output_tokens"}.
        """
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx(),
                "num_predict": chat_num_predict(),
            },
        }
        if tools:
            payload["tools"] = tools

        raw = self._post_json(url, payload)
        message = raw.get("message", {}) or {}
        content = message.get("content", "") or ""

        tool_calls: List[Dict[str, Any]] = []
        for tc in message.get("tool_calls", []) or []:
            fn = tc.get("function", {}) or {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append({"name": fn.get("name", ""), "arguments": args or {}})

        return {
            "content": content,
            "tool_calls": tool_calls,
            "input_tokens": int(raw.get("prompt_eval_count", 0)),
            "output_tokens": int(raw.get("eval_count", 0)),
        }
