from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone

from prompts import GENERATOR_SYSTEM_PROMPT, random_prompt

# ── Model lists ────────────────────────────────────────────────────────────────
MODELS = {
    "openai":    ["gpt-4o-mini", "gpt-4o"],
    "anthropic": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
    "gemini":    ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
}

# ── Cost table (USD per 1 000 tokens, rough estimates) ─────────────────────────
COST_PER_1K = {
    "gpt-4o-mini":                  0.00015,
    "gpt-4o":                       0.005,
    "claude-3-5-haiku-20241022":    0.00025,
    "claude-3-5-sonnet-20241022":   0.003,
    "claude-3-opus-20240229":       0.015,
    "gemini-1.5-flash":             0.000075,
    "gemini-1.5-pro":               0.00125,
    "gemini-2.0-flash":             0.0001,
}

_AVG_TOKENS_PER_CONV = 300  # rough estimate for cost calculation


def estimate_cost(model_name: str, count: int) -> float:
    """Return estimated USD cost for *count* conversations."""
    per_k = COST_PER_1K.get(model_name, 0.001)
    return count * _AVG_TOKENS_PER_CONV * per_k / 1000


# ── Package availability check ────────────────────────────────────────────────
_INSTALL_HINTS = {
    "openai":    "pip install openai",
    "anthropic": "pip install anthropic",
    "gemini":    "pip install google-genai",
}


def _check_package(provider: str) -> tuple[bool, str]:
    """Return (True, '') if the provider package is importable, else (False, hint)."""
    try:
        if provider == "openai":
            import openai  # noqa: F401
        elif provider == "anthropic":
            import anthropic  # noqa: F401
        elif provider == "gemini":
            from google import genai  # noqa: F401
        else:
            return False, f"Ismeretlen provider: {provider}"
        return True, ""
    except ImportError:
        hint = _INSTALL_HINTS.get(provider, "")
        return False, f"Hiányzó csomag a '{provider}' providerhez. Telepítsd: {hint}"


# ── Client factory ─────────────────────────────────────────────────────────────
def get_client(provider: str, api_key: str):
    """Return an initialised API client for the given provider."""
    if not api_key or not api_key.strip():
        raise ValueError("API kulcs hiányzik")

    ok, msg = _check_package(provider)
    if not ok:
        raise ImportError(msg)

    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key)

    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=api_key)

    if provider == "gemini":
        from google import genai
        return genai.Client(api_key=api_key)

    raise ValueError(f"Ismeretlen provider: {provider}")


# ── API key validation ─────────────────────────────────────────────────────────
def validate_api_key(provider: str, api_key: str) -> tuple[bool, str]:
    """Test whether *api_key* is valid for *provider*.

    Returns (True, "OK") or (False, error_message).
    """
    try:
        if provider == "openai":
            client = get_client("openai", api_key)
            client.models.list()
            return True, "OK"

        if provider == "anthropic":
            client = get_client("anthropic", api_key)
            client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True, "OK"

        if provider == "gemini":
            from google import genai
            client = genai.Client(api_key=api_key)
            list(client.models.list())
            return True, "OK"

        return False, f"Ismeretlen provider: {provider}"

    except ValueError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)


# ── JSON extraction helper ─────────────────────────────────────────────────────
def _extract_json(text: str) -> dict | None:
    """Try to parse JSON from *text*, with a regex fallback."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: grab the first {...} block (handles markdown code fences etc.)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ── Single-conversation generator ─────────────────────────────────────────────
def generate_one(provider: str, api_key: str, model_name: str) -> tuple[dict, str] | tuple[None, str]:
    """Generate one training conversation.

    Returns (data_dict, "") on success or (None, error_message) on failure.
    """
    prompt = random_prompt()
    backoff_delays = [5, 10, 20]
    last_error = ""

    for attempt in range(3):
        try:
            raw = _call_api(provider, api_key, model_name, prompt)
            data = _extract_json(raw)

            if data is None:
                last_error = f"Nem sikerült JSON-t kinyerni a válaszból (attempt {attempt+1})"
                continue

            # Validate required fields
            if "user_message" not in data or "assistant_message" not in data:
                last_error = f"Hiányzó mezők a válaszban (attempt {attempt+1})"
                continue

            data.setdefault("tema", "ismeretlen téma")
            data["generalva"] = datetime.now(timezone.utc).isoformat()
            return data, ""

        except ImportError as exc:
            return None, f"Hiányzó csomag: {exc}"
        except Exception as exc:
            last_error = str(exc)
            err_lower = last_error.lower()
            is_rate_limit = any(
                kw in err_lower
                for kw in ("rate limit", "ratelimit", "429", "quota", "resource_exhausted")
            )
            delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
            if is_rate_limit:
                time.sleep(delay)
            elif attempt < 2:
                time.sleep(2)

    return None, last_error


# ── Provider-specific API calls ────────────────────────────────────────────────
def _call_api(provider: str, api_key: str, model_name: str, prompt: str) -> str:
    """Dispatch to the correct provider and return raw response text."""
    if provider == "openai":
        return _call_openai(api_key, model_name, prompt)
    if provider == "anthropic":
        return _call_anthropic(api_key, model_name, prompt)
    if provider == "gemini":
        return _call_gemini(api_key, model_name, prompt)
    raise ValueError(f"Ismeretlen provider: {provider}")


def _call_openai(api_key: str, model_name: str, prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    system_msg = {"role": "system", "content": GENERATOR_SYSTEM_PROMPT}
    user_msg   = {"role": "user",   "content": prompt}

    response = client.chat.completions.create(
        model=model_name,
        messages=[system_msg, user_msg],
        max_tokens=400,
        temperature=0.85,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _call_anthropic(api_key: str, model_name: str, prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    system = GENERATOR_SYSTEM_PROMPT + "\nFONTOS: Csak JSON-t adj vissza."

    response = client.messages.create(
        model=model_name,
        max_tokens=400,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text if response.content else ""


def _call_gemini(api_key: str, model_name: str, prompt: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=GENERATOR_SYSTEM_PROMPT,
            response_mime_type="application/json",
            max_output_tokens=400,
            temperature=0.85,
        ),
    )
    return response.text or ""
