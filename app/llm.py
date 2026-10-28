import openai
from openai import OpenAI
from pydantic import ValidationError
from app.config import (
    GOOGLE_API_KEY, LLM_BASE_URL, MODEL, LLM_TIMEOUT_SECONDS, LLM_INVALID_OUTPUT_RETRIES,
)
from app.schemas import LLMAnswer

# Talking to the LLM: Gemini through Google's OpenAI-compatible endpoint (same as the scripts).
#
# Two kinds of things can go wrong here, and both are HANDLED, never a crash:
#   1. The provider fails (rate limit 429, timeout, network, bad key, server error)
#      -> we raise ProviderError with a message for the user and an HTTP status for the API.
#   2. The provider answers, but the text is not valid JSON / does not match LLMAnswer
#      -> we retry once, then raise InvalidOutputError.


class ProviderError(Exception):
    def __init__(self, message, http_status):
        super().__init__(message)
        self.message = message
        self.http_status = http_status


class InvalidOutputError(Exception):
    pass


# No API key = no client. We do not crash at startup: the rest of the app (documents,
# retrieval, evaluation) works without the LLM, and /api/ask explains what is missing.
if GOOGLE_API_KEY:
    # max_retries=0: the SDK would otherwise silently retry 2 times on 429/5xx.
    # We prefer to tell the user right away, and keep the number of calls predictable.
    client = OpenAI(
        api_key=GOOGLE_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=0,
    )
else:
    client = None
    print("WARNING: GOOGLE_API_KEY is not set (see .env.example). /api/ask will return an error.")


def call_llm(prompt):
    if client is None:
        raise ProviderError("The LLM is not configured: set GOOGLE_API_KEY in .env and restart.", 500)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            # JSON mode: asks the model for a JSON object. It does NOT guarantee our schema,
            # that is why we still validate the result with Pydantic below.
            response_format={"type": "json_object"},
        )
    # The order matters: APITimeoutError is a subclass of APIConnectionError.
    except openai.RateLimitError:
        raise ProviderError("The LLM provider is rate limiting us (429). Wait a few seconds and retry.", 429)
    except openai.APITimeoutError:
        raise ProviderError(f"The LLM provider did not answer within {LLM_TIMEOUT_SECONDS}s. Retry.", 504)
    except openai.APIConnectionError:
        raise ProviderError("Could not reach the LLM provider (network problem).", 502)
    except openai.AuthenticationError:
        raise ProviderError("The LLM provider rejected the API key. Check GOOGLE_API_KEY in .env.", 500)
    except openai.NotFoundError:
        # Models get retired: a 404 here usually means MODEL in app/config.py no longer exists.
        raise ProviderError(f"The LLM provider does not know the model {MODEL!r} (404). Change MODEL in app/config.py.", 500)
    except openai.APIStatusError as e:
        raise ProviderError(f"The LLM provider returned an error (HTTP {e.status_code}).", 502)

    return response.choices[0].message.content or ""


def ask_llm_for_answer(prompt):
    # 1 normal attempt + LLM_INVALID_OUTPUT_RETRIES extra attempts if the output is invalid.
    for attempt in range(1 + LLM_INVALID_OUTPUT_RETRIES):
        raw = call_llm(prompt)
        try:
            return LLMAnswer.model_validate_json(raw)
        except ValidationError as e:
            # The raw text stays in the server log (useful to debug), it is never shown as an answer.
            print(f"Invalid LLM output (attempt {attempt + 1}): {e.error_count()} error(s). Raw: {raw[:500]!r}")

    raise InvalidOutputError()
