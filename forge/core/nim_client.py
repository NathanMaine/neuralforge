"""NIM inference client -- OpenAI-compatible for TensorRT-LLM."""

import json
import logging
import re

import httpx

import forge.config as config

log = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


async def chat_completion(
    messages: list[dict],
    max_tokens: int = 1000,
    temperature: float = 0.7,
    stream: bool = False,
) -> dict | None:
    """POST to NIM /v1/chat/completions.

    Returns the full response dict on success, or None on error.
    """
    url = f"{config.NIM_URL}/v1/chat/completions"
    payload = {
        "model": config.NIM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                log.warning("NIM returned empty response body")
                return None
            return data
    except httpx.TimeoutException:
        log.error("NIM request timed out: %s", url)
        return None
    except httpx.HTTPStatusError as exc:
        log.error("NIM HTTP %s: %s", exc.response.status_code, exc.response.text)
        return None
    except httpx.ConnectError:
        log.error("NIM connection refused: %s", url)
        return None
    except Exception:
        log.exception("Unexpected NIM error")
        return None


def _extract_json(text: str) -> dict | None:
    """Parse JSON from plain text or markdown-wrapped ```json ... ``` blocks."""
    # Try plain JSON first
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Try markdown-fenced JSON
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    return None


async def classify_json(
    prompt: str,
    max_retries: int = 2,
) -> dict | None:
    """Send *prompt* to NIM and parse the JSON response.

    Retries up to *max_retries* times when the model returns malformed JSON.
    Handles markdown-wrapped JSON (``````json ... ``````).
    """
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(1, max_retries + 1):
        result = await chat_completion(messages, temperature=0.2)
        if result is None:
            return None

        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            log.warning("Malformed NIM response structure (attempt %d/%d)", attempt, max_retries)
            continue

        parsed = _extract_json(content)
        if parsed is not None:
            return parsed

        log.warning(
            "Failed to parse JSON from NIM response (attempt %d/%d): %.120s",
            attempt,
            max_retries,
            content,
        )

    log.error("All %d JSON-parse retries exhausted", max_retries)
    return None


async def stream_completion(
    messages: list[dict],
    max_tokens: int = 1000,
):
    """Async generator -- yields SSE data lines from a streaming NIM response.

    Each yielded value is the parsed JSON dict from one ``data: ...`` SSE line.
    The final ``data: [DONE]`` sentinel is not yielded.
    """
    url = f"{config.NIM_URL}/v1/chat/completions"
    payload = {
        "model": config.NIM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str == "[DONE]":
                        return
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        log.warning("Unparseable SSE chunk: %.120s", data_str)
    except httpx.TimeoutException:
        log.error("NIM stream timed out: %s", url)
    except httpx.HTTPStatusError as exc:
        log.error("NIM stream HTTP %s: %s", exc.response.status_code, exc.response.text)
    except httpx.ConnectError:
        log.error("NIM stream connection refused: %s", url)
    except Exception:
        log.exception("Unexpected NIM stream error")
