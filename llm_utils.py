"""
llm_utils.py
One place for LLM-call error handling, so every call site behaves the same way.

Transient failures (rate limits, timeouts, connection drops, 5xx) are retried
with exponential backoff. Permanent failures (bad key, unknown model, bad
request) fail immediately, because retrying them only wastes time.
"""

import time

# HTTP status codes where trying again cannot help
NON_RETRYABLE_STATUS = {400, 401, 403, 404}


class LLMCallError(RuntimeError):
    """Raised when an LLM call fails for good (retries exhausted or non-retryable)."""


def safe_invoke(llm, messages, retries: int = 3, base_delay: float = 1.0, sleep=time.sleep):
    """Call llm.invoke(messages), retrying transient failures with backoff.

    `sleep` is injectable so tests can run instantly.
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as err:  # noqa: BLE001 - SDK raises many exception types
            last_error = err
            status = getattr(err, "status_code", None)
            if status in NON_RETRYABLE_STATUS:
                raise LLMCallError(f"LLM call failed with non-retryable status {status}: {err}") from err
            if attempt < retries:
                sleep(base_delay * (2 ** (attempt - 1)))  # 1s, 2s, 4s...
    raise LLMCallError(f"LLM call failed after {retries} attempts: {last_error}") from last_error
