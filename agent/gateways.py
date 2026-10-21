"""Official contest gateway routing without forwarding keys to a new service.

The organizer's two hosts share one account and quota. A custom configured
endpoint remains custom: its credentials are never sent to the contest hosts.
Only the request coordinator may retry, so each network attempt remains charged
to its existing request, token and wall-clock guards.
"""

from __future__ import annotations

from urllib.parse import urlsplit


ALTERNATE_BASE_URL = "https://api.alfahack-gateway26.ru/v1"
PRIMARY_BASE_URL = "https://changellenge.up.railway.app/v1"
OFFICIAL_GATEWAYS = (("alternate", ALTERNATE_BASE_URL), ("primary", PRIMARY_BASE_URL))
CONNECT_TIMEOUT_SECONDS = 10.0


def official_gateway_name(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value.strip())
        if (parsed.scheme != "https" or parsed.username is not None
                or parsed.password is not None or parsed.port not in {None, 443}
                or parsed.query or parsed.fragment or parsed.path.rstrip("/") != "/v1"):
            return None
    except ValueError:
        return None
    return next((name for name, url in OFFICIAL_GATEWAYS
                 if parsed.hostname == urlsplit(url).hostname), None)


def configured_gateways(base_url):
    """Prefer the new official host; preserve every custom address verbatim."""
    if not isinstance(base_url, str) or not base_url.strip():
        return ()
    if official_gateway_name(base_url) is not None:
        return OFFICIAL_GATEWAYS
    return (("custom", base_url),)


def client_gateways(client):
    """Fake/embedded clients without an endpoint retain their original API."""
    value = getattr(client, "base_url", None)
    if value is None:
        return ()
    if not isinstance(value, str):
        # The SDK stores httpx.URL. Do not stringify arbitrary mock objects.
        try:
            import httpx
        except ImportError:
            return ()
        if not isinstance(value, httpx.URL):
            return ()
        value = str(value)
    return configured_gateways(value)


def request_options(timeout, route=None):
    if route is None:
        return {"timeout": timeout}
    import httpx

    return {"base_url": route[1],
            "timeout": httpx.Timeout(timeout, connect=min(CONNECT_TIMEOUT_SECONDS, timeout))}


def retryable_api_error(error):
    """Retry typed transport/overload failures, never shared account exhaustion."""
    from openai import APIConnectionError, APIStatusError

    fields = [getattr(error, name, None) for name in ("code", "type")]
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        fields.extend(body.get(name) for name in ("code", "type"))
        nested = body.get("error")
        if isinstance(nested, dict):
            fields.extend(nested.get(name) for name in ("code", "type"))
    exhausted = ("quota", "balance", "billing", "budget", "credit", "fund", "payment")
    if any(isinstance(value, str) and any(word in value.lower() for word in exhausted)
           for value in fields):
        return False
    if isinstance(error, APIConnectionError):
        return True
    return (isinstance(error, APIStatusError)
            and (error.status_code == 429 or 500 <= error.status_code <= 599))
