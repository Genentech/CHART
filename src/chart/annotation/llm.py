"""Model plumbing for the archetype annotation pipeline.

Two providers are supported and the pipeline runs identically on either:

  gateway  OpenAI-compatible LiteLLM gateway (default when its vars are set)
               export LITELLM_API_KEY="sk-..."
               export LITELLM_BASE_URL="https://<gateway-host>"   # '/v1' appended if absent
           OPENAI_API_KEY / OPENAI_BASE_URL are accepted as fallbacks.
           OPENAI_API_KEY alone, with no base URL, goes to OpenAI itself.

  azure    Azure OpenAI
               export AZURE_OPENAI_API_KEY="..."
               export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
               export AZURE_OPENAI_API_VERSION="2025-03-01-preview"     # optional
           API_KEY_SC / API_KEY_EUS2 are accepted as fallbacks, matching the
           older scripts. Deployment names default to the model name; override
           with AZURE_DEPLOYMENT_MAP='{"gpt-5.5": "gpt-5-5-deploy"}'.

Pick one with LLM_PROVIDER=gateway|azure|auto.
`auto` (the default) prefers the gateway and falls back to Azure.

No credentials are hardcoded here: everything comes from the environment.

The other rule this module enforces: an LLM call either returns usable text or
raises. Empty replies, replies truncated by the token budget, and replies
missing their required section headers are errors, so no step can write a
silently degenerate file and report success.
"""

import base64
import importlib.util
import io
import json
import logging
import mimetypes
import os
import random
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

try:
    import tiktoken
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
except ImportError:
    BaseCallbackHandler = object
    AzureChatOpenAI = ChatOpenAI = None

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelProfile:
    """What the pipeline needs to know about a model.

    reasoning        takes reasoning_effort/verbosity instead of temperature
    azure_deployment default Azure deployment name (AZURE_DEPLOYMENT_MAP wins)
    rates            USD per 1K tokens, or None when pricing is not configured
    """
    reasoning: bool
    azure_deployment: str
    rates: dict = None


MODEL_PROFILES = {
    "gpt-5":   ModelProfile(reasoning=True,  azure_deployment="gpt-5"),
    "gpt-5.5": ModelProfile(reasoning=True,  azure_deployment="gpt-5.5"),
    "gpt-4o":  ModelProfile(reasoning=False, azure_deployment="gpt-4o",
                            rates={"prompt": 0.0025, "completion": 0.01}),
}

DEFAULT_MODEL = "gpt-5"

# Reasoning tokens are billed against max_completion_tokens, so a high-effort
# call with a small budget can spend the whole budget thinking and return an
# empty string. Keep this generous; each step sets its own budget on top.
DEFAULT_MAX_COMPLETION_TOKENS = 8000


def model_profile(model: str) -> ModelProfile:
    """Registry entry for a model, or a guess for one that is not registered."""
    if model in MODEL_PROFILES:
        return MODEL_PROFILES[model]
    # Unregistered names still work; infer from the family prefix so a new
    # gateway model can be used without editing this file.
    reasoning = bool(re.match(r"^(gpt-5|o[134])", model))
    logging.info("Model %s is not in MODEL_PROFILES; assuming reasoning=%s", model, reasoning)
    return ModelProfile(reasoning=reasoning, azure_deployment=model)


def model_supports_temperature(model: str) -> bool:
    """GPT-5 style models take reasoning_effort/verbosity instead."""
    return not model_profile(model).reasoning


def gpt5_reasoning_effort() -> str:
    return os.environ.get("GPT5_REASONING_EFFORT", "medium")


def gpt5_verbosity() -> str:
    return os.environ.get("GPT5_VERBOSITY", "medium")


def gpt5_max_completion_tokens() -> int:
    return int(os.environ.get("GPT5_MAX_COMPLETION_TOKENS", str(DEFAULT_MAX_COMPLETION_TOKENS)))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class PipelineError(RuntimeError):
    """Base class for every failure this pipeline raises on purpose."""


class ProviderConfigError(PipelineError):
    """Provider credentials are missing or unusable."""


class EmptyResponseError(PipelineError):
    """The model returned no text."""


class TruncatedResponseError(PipelineError):
    """The model hit the completion-token ceiling mid-answer."""


class ResponseFormatError(PipelineError):
    """The reply is missing a section header the next step depends on."""


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

PROVIDER_GATEWAY = "gateway"
PROVIDER_AZURE = "azure"


def _first_env(names):
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return name, value
    return None, ""


class GatewayConfig:
    """Credentials for the OpenAI-compatible LiteLLM gateway."""

    API_KEY_VARS = ("LITELLM_API_KEY", "OPENAI_API_KEY")
    BASE_URL_VARS = ("LITELLM_BASE_URL", "LITELLM_API_BASE", "OPENAI_BASE_URL")

    @staticmethod
    def available() -> bool:
        return bool(_first_env(GatewayConfig.API_KEY_VARS)[1]
                    and _first_env(GatewayConfig.BASE_URL_VARS)[1]
                    or os.environ.get("OPENAI_API_KEY", "").strip())

    @staticmethod
    def _ensure_v1(base_url: str) -> str:
        """The gateway exposes the OpenAI routes under /v1."""
        trimmed = base_url.rstrip("/")
        if re.search(r"/v\d+$", trimmed):
            return trimmed
        return f"{trimmed}/v1"

    @staticmethod
    def get_credentials(model_type: str = ""):
        _, api_key = _first_env(GatewayConfig.API_KEY_VARS)
        if not api_key:
            raise ProviderConfigError(
                "Missing gateway API key. Set one of: "
                + ", ".join(GatewayConfig.API_KEY_VARS)
                + '  (e.g. export LITELLM_API_KEY="sk-...")'
            )

        var_name, base_url = _first_env(GatewayConfig.BASE_URL_VARS)
        if not base_url:
            if os.environ.get("OPENAI_API_KEY", "").strip():
                return os.environ["OPENAI_API_KEY"].strip(), None
            raise ProviderConfigError(
                "Missing gateway base URL. Set one of: "
                + ", ".join(GatewayConfig.BASE_URL_VARS)
                + '  (e.g. export LITELLM_BASE_URL="https://<gateway-host>"), '
                "or set OPENAI_API_KEY alone to use OpenAI itself"
            )

        base_url = GatewayConfig._ensure_v1(base_url)
        logging.debug("Using gateway %s (%s) for model %s", base_url, var_name, model_type)
        return api_key, base_url


class AzureConfig:
    """Credentials for Azure OpenAI. Nothing is hardcoded; set the env vars."""

    API_KEY_VARS = ("AZURE_OPENAI_API_KEY", "API_KEY_SC", "API_KEY_EUS2")
    ENDPOINT_VARS = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_ENDPOINT_SC")
    DEFAULT_API_VERSION = "2025-03-01-preview"

    @staticmethod
    def available() -> bool:
        return bool(_first_env(AzureConfig.API_KEY_VARS)[1]
                    and _first_env(AzureConfig.ENDPOINT_VARS)[1])

    @staticmethod
    def api_version() -> str:
        return (os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()
                or os.environ.get("OPENAI_API_VERSION", "").strip()
                or AzureConfig.DEFAULT_API_VERSION)

    @staticmethod
    def deployment_for(model: str) -> str:
        """Azure addresses deployments, not model names.

        AZURE_DEPLOYMENT_MAP is a JSON object, e.g.
            export AZURE_DEPLOYMENT_MAP='{"gpt-5.5": "gpt-5-5-preview"}'
        """
        raw = os.environ.get("AZURE_DEPLOYMENT_MAP", "").strip()
        if raw:
            try:
                overrides = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ProviderConfigError(
                    f"AZURE_DEPLOYMENT_MAP is not valid JSON: {exc}"
                ) from exc
            if model in overrides:
                return str(overrides[model])
        return model_profile(model).azure_deployment

    @staticmethod
    def get_credentials(model: str):
        _, api_key = _first_env(AzureConfig.API_KEY_VARS)
        if not api_key:
            raise ProviderConfigError(
                "Missing Azure OpenAI API key. Set one of: "
                + ", ".join(AzureConfig.API_KEY_VARS)
                + '  (e.g. export AZURE_OPENAI_API_KEY="...")'
            )

        _, endpoint = _first_env(AzureConfig.ENDPOINT_VARS)
        if not endpoint:
            raise ProviderConfigError(
                "Missing Azure OpenAI endpoint. Set one of: "
                + ", ".join(AzureConfig.ENDPOINT_VARS)
                + '  (e.g. export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/")'
            )

        return api_key, endpoint.rstrip("/") + "/", AzureConfig.api_version(), AzureConfig.deployment_for(model)


def resolve_provider(requested: str = None) -> str:
    """Which provider to use: explicit argument, then LLM_PROVIDER, then auto."""
    choice = (requested or os.environ.get("LLM_PROVIDER") or "auto").strip().lower()

    if choice in {"gateway", "litellm"}:
        if not GatewayConfig.available():
            GatewayConfig.get_credentials()  # raises with the specific missing var
        return PROVIDER_GATEWAY

    if choice in {"azure", "azure_openai", "azureopenai"}:
        if not AzureConfig.available():
            AzureConfig.get_credentials(DEFAULT_MODEL)
        return PROVIDER_AZURE

    if choice != "auto":
        raise ProviderConfigError(
            f"Unknown LLM provider '{choice}'. Use one of: gateway, azure, auto."
        )

    if GatewayConfig.available():
        return PROVIDER_GATEWAY
    if AzureConfig.available():
        return PROVIDER_AZURE

    raise ProviderConfigError(
        "No LLM provider is configured, and only the annotation stage needs one. Set one of\n"
        '  openai:   export OPENAI_API_KEY="sk-..."\n'
        '  gateway:  export LITELLM_API_KEY="sk-..." '
        'and LITELLM_BASE_URL="https://<gateway-host>"\n'
        '  azure:    export AZURE_OPENAI_API_KEY="..." '
        'and AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"\n'
        "then optionally pin one with LLM_PROVIDER=gateway|azure."
    )


def describe_provider(provider: str = None, model: str = DEFAULT_MODEL) -> str:
    """One-line provider summary for logs, with no secrets in it."""
    provider = resolve_provider(provider)
    if provider == PROVIDER_AZURE:
        _, endpoint, api_version, deployment = AzureConfig.get_credentials(model)
        return f"azure {endpoint} (deployment={deployment}, api_version={api_version})"
    _, base_url = GatewayConfig.get_credentials(model)
    return f"gateway {base_url}" if base_url else "openai"


def check_llm_credentials(provider: str = None, model: str = DEFAULT_MODEL) -> str:
    """Preflight so a run dies in the first second, not after step 1."""
    missing = [name for name in ("langchain_openai", "sentence_transformers")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise ProviderConfigError(
            f"Missing {', '.join(missing)}. Install them with pip install '.[annotation]' "
            "in the CHART directory; only the annotation stage needs them."
        )
    return describe_provider(provider, model)


# ---------------------------------------------------------------------------
# Cost / token accounting
# ---------------------------------------------------------------------------

class CostCallback(BaseCallbackHandler):
    def __init__(self, model: str):
        self.model = model
        self.rates = model_profile(model).rates
        try:
            self.enc = tiktoken.encoding_for_model(model)
        except Exception:
            self.enc = tiktoken.get_encoding("cl100k_base")
        self.total_cost = 0.0
        self.total_tokens = {"prompt": 0, "completion": 0, "reasoning": 0}
        self.calls = 0
        self.last_prompt = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        if prompts:
            self.last_prompt = prompts[0]

    def on_llm_end(self, response, **kwargs):
        self.calls += 1
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)

        if prompt_tokens == 0 and self.last_prompt:
            prompt_tokens = len(self.enc.encode(self.last_prompt))
        if completion_tokens == 0:
            completion_tokens = sum(
                len(self.enc.encode(gen.text))
                for gens in response.generations
                for gen in gens
            )

        self.total_tokens["prompt"] += prompt_tokens
        self.total_tokens["completion"] += completion_tokens
        self.total_tokens["reasoning"] += reasoning_tokens or 0

        if self.rates:
            self.total_cost += (prompt_tokens * self.rates["prompt"]
                                + completion_tokens * self.rates["completion"]) / 1000

    def report(self) -> str:
        parts = [
            f"calls={self.calls}",
            f"prompt={self.total_tokens['prompt']}",
            f"completion={self.total_tokens['completion']}",
        ]
        if self.total_tokens["reasoning"]:
            parts.append(f"reasoning={self.total_tokens['reasoning']}")
        if self.rates:
            parts.append(f"cost=${self.total_cost:.4f}")
        return "  ".join(parts)


def make_llm(
    model: str = DEFAULT_MODEL,
    temp: float = 0.5,
    streaming: bool = False,
    *,
    provider: str = None,
    reasoning_effort: str = None,
    verbosity: str = None,
    max_completion_tokens: int = None,
    **kwargs,
):
    """Build a chat model on the selected provider, with a CostCallback attached."""
    provider = resolve_provider(provider)
    profile = model_profile(model)

    llm_kwargs = {
        "callbacks": [CostCallback(model)],
        "streaming": streaming,
        **kwargs,
    }

    if not profile.reasoning:
        llm_kwargs["temperature"] = temp
    else:
        budget = int(max_completion_tokens or gpt5_max_completion_tokens())
        llm_kwargs["reasoning_effort"] = reasoning_effort or gpt5_reasoning_effort()
        llm_kwargs["verbosity"] = verbosity or gpt5_verbosity()
        llm_kwargs["max_completion_tokens"] = budget
        logging.info(
            "%s on %s: reasoning_effort=%s verbosity=%s max_completion_tokens=%s "
            "(temperature=%s ignored)",
            model, provider, llm_kwargs["reasoning_effort"],
            llm_kwargs["verbosity"], budget, temp,
        )

    if provider == PROVIDER_AZURE:
        api_key, endpoint, api_version, deployment = AzureConfig.get_credentials(model)
        with warnings.catch_warnings():
            # AzureChatOpenAI routes max_completion_tokens through model_kwargs
            # and warns about it; the parameter still reaches the API.
            warnings.filterwarnings("ignore", message=".*max_completion_tokens.*")
            return AzureChatOpenAI(
                azure_endpoint=endpoint,
                azure_deployment=deployment,
                api_key=api_key,
                api_version=api_version,
                max_retries=6,
                **llm_kwargs,
            )

    api_key, base_url = GatewayConfig.get_credentials(model)
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, max_retries=6, **llm_kwargs)


def cost_callback_of(llm):
    """The CostCallback attached by make_llm, or None.

    The chat models are pydantic and reject extra attributes, so the handler is
    found by scanning the callback list rather than stashed on the object.
    """
    callbacks = getattr(llm, "callbacks", None) or []
    return next((cb for cb in callbacks if isinstance(cb, CostCallback)), None)


# ---------------------------------------------------------------------------
# Calling the model
# ---------------------------------------------------------------------------

def invoke_with_retry(llm, payload, *, max_attempts: int = 6, base_delay: float = 8.0):
    """Invoke with exponential backoff on rate limits. Other errors propagate."""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(payload)
        except Exception as exc:
            last_exc = exc
            message = str(exc)
            is_rate_limited = (
                "429" in message
                or "RateLimit" in type(exc).__name__
                or "Too Many Requests" in message
            )
            if not is_rate_limited or attempt == max_attempts:
                raise

            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1.5)
            logging.warning(
                "Rate limited (attempt %s/%s). Sleeping %.1fs before retrying.",
                attempt, max_attempts, delay,
            )
            time.sleep(delay)

    raise last_exc


def response_text(response) -> str:
    """Flatten a chat response to plain text across content shapes."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()
    return str(content).strip()


def _finish_reason(response) -> str:
    meta = getattr(response, "response_metadata", None) or {}
    return meta.get("finish_reason") or ""


def _reasoning_tokens(response) -> int:
    usage = getattr(response, "usage_metadata", None) or {}
    details = usage.get("output_token_details") or {}
    return int(details.get("reasoning") or 0)


def _budget_hint(llm) -> str:
    budget = getattr(llm, "max_tokens", None)
    if budget is None:
        budget = (getattr(llm, "model_kwargs", None) or {}).get("max_completion_tokens")
    effort = getattr(llm, "reasoning_effort", None)
    return f"max_completion_tokens={budget}, reasoning_effort={effort}"


def _missing_sections(text: str, require) -> list:
    lowered = text.lower()
    return [marker for marker in require if marker.lower() not in lowered]


def invoke_text(
    llm,
    messages,
    *,
    context: str,
    require=(),
    min_chars: int = 40,
    format_retries: int = 1,
) -> str:
    """Call the model and return validated text, or raise.

    context         short label used in error messages, e.g. "step3 summary A7"
    require         section headers that must appear in the reply
    min_chars       shortest reply considered real
    format_retries  re-asks once (by default) when a required header is missing
    """
    attempt = 0
    while True:
        response = invoke_with_retry(llm, messages)
        text = response_text(response)
        finish = _finish_reason(response)
        reasoning = _reasoning_tokens(response)

        if not text:
            detail = f"finish_reason={finish or 'unknown'}"
            if reasoning:
                detail += f", reasoning_tokens={reasoning}"
            raise EmptyResponseError(
                f"{context}: model returned no text ({detail}; {_budget_hint(llm)}). "
                "The completion budget is most likely being consumed by reasoning "
                "tokens - raise max_completion_tokens or lower reasoning_effort."
            )

        if finish == "length":
            raise TruncatedResponseError(
                f"{context}: reply was cut off at the token ceiling "
                f"({_budget_hint(llm)}, {len(text)} chars returned). "
                "Raise max_completion_tokens."
            )

        if len(text) < min_chars:
            raise EmptyResponseError(
                f"{context}: reply is implausibly short ({len(text)} chars, "
                f"minimum {min_chars}): {text!r}"
            )

        missing = _missing_sections(text, require)
        if not missing:
            return text

        if attempt < format_retries:
            attempt += 1
            logging.warning(
                "%s: reply missing %s - re-asking (attempt %s/%s)",
                context, ", ".join(missing), attempt, format_retries,
            )
            messages = list(messages) + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "Your reply is missing these required section headers: "
                        + ", ".join(missing)
                        + ". Send the complete answer again, using exactly those headers."
                    ),
                },
            ]
            continue

        raise ResponseFormatError(
            f"{context}: reply is missing required section header(s) "
            f"{', '.join(missing)} after {attempt + 1} attempt(s). "
            f"First 300 chars: {text[:300]!r}"
        )


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

MIN_RESOLUTION = (512, 512)

ANNOTATION_SYSTEM_PROMPT = (
    "You are a careful microscopy image analyst. When given an image, describe "
    "only what is visible without guessing or bringing in prior biological "
    "knowledge. Use precise and observational language."
)


class ImageError(PipelineError):
    """An input image is missing, unreadable, or unusable."""


def verify_image(image_path) -> Tuple[int, int]:
    """Return (width, height); raise if the file is missing or unreadable."""
    path = Path(image_path)
    if not path.exists():
        raise ImageError(f"Image not found: {path}")
    try:
        with Image.open(path) as img:
            width, height = img.size
    except Exception as exc:
        raise ImageError(f"Could not read image {path}: {exc}") from exc

    if width < MIN_RESOLUTION[0] or height < MIN_RESOLUTION[1]:
        logging.warning(
            "Image %s is below the %sx%s resolution guide (%sx%s)",
            path.name, MIN_RESOLUTION[0], MIN_RESOLUTION[1], width, height,
        )
    return width, height


def encode_image(image_path) -> Tuple[str, str]:
    """Base64-encode an image at full resolution, with its MIME type."""
    path = Path(image_path)
    if not path.exists():
        raise ImageError(f"Image not found: {path}")
    data = path.read_bytes()
    mime_type, _ = mimetypes.guess_type(str(path))
    return base64.b64encode(data).decode(), mime_type or "image/png"


def image_block(image_path, detail: str = "high", max_edge: Optional[int] = None) -> dict:
    """An OpenAI-style image_url content block.

    max_edge downsamples first, which is what the multi-image comparison steps
    use to keep a handful of grids inside one request.
    """
    path = Path(image_path)
    if not path.exists():
        raise ImageError(f"Image not found: {path}")

    if max_edge:
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((max_edge, max_edge))
                buffer = io.BytesIO()
                img.save(buffer, format="PNG", optimize=True)
                encoded = base64.b64encode(buffer.getvalue()).decode()
        except Exception as exc:
            raise ImageError(f"Could not downsample image {path}: {exc}") from exc
        mime_type = "image/png"
    else:
        encoded, mime_type = encode_image(path)

    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}", "detail": detail},
    }


def analyze_image(
    llm,
    image_path,
    prompt: str,
    *,
    context: str,
    require=(),
    detail: str = "high",
) -> str:
    """Send one image plus a prompt and return validated text."""
    messages = [
        {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.strip()},
                image_block(image_path, detail=detail),
            ],
        },
    ]
    return invoke_text(llm, messages, context=context, require=require)
