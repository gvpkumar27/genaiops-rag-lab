import time

import requests

from app.config import settings

_RETRY_STATUSES = {429, 500, 502, 503, 504}


def post_json(
    url: str,
    payload: dict,
    timeout: int,
) -> requests.Response:
    """POST JSON with bounded retry/backoff for transient failures."""
    attempts = max(1, settings.HTTP_RETRIES + 1)
    backoff = max(0.0, settings.HTTP_RETRY_BACKOFF_SEC)
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            if response.status_code in _RETRY_STATUSES and attempt < attempts:
                time.sleep(backoff * attempt)
                continue
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
            if attempt >= attempts:
                raise
            time.sleep(backoff * attempt)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unexpected retry state in post_json")
