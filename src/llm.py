from dataclasses import dataclass
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import LLMConfig

SUPPORTED_PROVIDERS = {"openai_compatible", "anthropic"}


class LLMConfigurationError(ValueError):
    """Raised when the LLM configuration is incomplete or unsupported."""


@dataclass
class LLMGenerationResult:
    """Generated answer returned by the configured LLM provider."""

    answer: str
    model_name: str


def build_chat_completions_url(base_url: str) -> str:
    """Build an OpenAI-compatible chat completions URL from a base URL."""
    cleaned_base_url = base_url.strip().rstrip("/")

    if not cleaned_base_url:
        raise LLMConfigurationError("LLM_BASE_URL is not configured.")

    if cleaned_base_url.endswith("/chat/completions"):
        return cleaned_base_url

    return f"{cleaned_base_url}/chat/completions"


def build_anthropic_messages_url(base_url: str) -> str:
    """Build an Anthropic Messages API URL from a configurable base URL."""
    cleaned_base_url = base_url.strip().rstrip("/")

    if not cleaned_base_url:
        raise LLMConfigurationError("LLM_BASE_URL is not configured.")

    if cleaned_base_url.endswith("/messages"):
        return cleaned_base_url

    return f"{cleaned_base_url}/messages"


def validate_llm_config(llm_config: LLMConfig) -> None:
    """Validate the configured LLM provider before generation."""
    if llm_config.provider not in SUPPORTED_PROVIDERS:
        raise LLMConfigurationError(
            f"Unsupported LLM provider: {llm_config.provider}. "
            f"Supported providers: {', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )

    if not llm_config.api_key.strip():
        raise LLMConfigurationError("LLM_API_KEY is not configured.")

    if not llm_config.base_url.strip():
        raise LLMConfigurationError("LLM_BASE_URL is not configured.")

    if not llm_config.model_name.strip() or llm_config.model_name == "your-llm-model":
        raise LLMConfigurationError("LLM model_name is not configured.")


def generate_answer(prompt: str, llm_config: LLMConfig) -> LLMGenerationResult:
    """Generate an answer using the configured provider protocol."""
    validate_llm_config(llm_config)

    if llm_config.provider == "openai_compatible":
        return _generate_openai_compatible(prompt, llm_config)

    if llm_config.provider == "anthropic":
        return _generate_anthropic(prompt, llm_config)

    raise LLMConfigurationError(f"Unsupported LLM provider: {llm_config.provider}.")


def _post_json(url: str, body: dict, headers: dict[str, str]) -> dict:
    """Post JSON and return a decoded response body."""
    encoded_body = json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=encoded_body,
        headers=headers | {"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("LLM request timed out.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM response was not valid JSON.") from exc


def _generate_openai_compatible(prompt: str, llm_config: LLMConfig) -> LLMGenerationResult:
    """Generate an answer with an OpenAI-compatible chat completions API."""
    request_body = {
        "model": llm_config.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": llm_config.temperature,
        "max_tokens": llm_config.max_tokens,
    }
    response_body = _post_json(
        build_chat_completions_url(llm_config.base_url),
        request_body,
        {
            "Authorization": f"Bearer {llm_config.api_key}",
        },
    )

    try:
        answer = response_body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenAI-compatible response did not contain a chat answer.") from exc

    if not answer:
        raise RuntimeError("LLM returned an empty answer.")

    return LLMGenerationResult(answer=answer, model_name=llm_config.model_name)


def _generate_anthropic(prompt: str, llm_config: LLMConfig) -> LLMGenerationResult:
    """Generate an answer with the Anthropic Messages API."""
    request_body = {
        "model": llm_config.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": llm_config.temperature,
        "max_tokens": llm_config.max_tokens,
    }
    response_body = _post_json(
        build_anthropic_messages_url(llm_config.base_url),
        request_body,
        {
            "x-api-key": llm_config.api_key,
            "anthropic-version": "2023-06-01",
        },
    )

    content_blocks = response_body.get("content", [])
    if not isinstance(content_blocks, list):
        raise RuntimeError("Anthropic response content was not a list.")

    text_parts: list[str] = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "").strip()
            if text:
                text_parts.append(text)

    # Future tool-use blocks can appear in this array; Phase 3 only handles text.
    answer = "\n".join(text_parts).strip()
    if not answer:
        raise RuntimeError("Anthropic response did not contain a text answer.")

    return LLMGenerationResult(answer=answer, model_name=llm_config.model_name)
