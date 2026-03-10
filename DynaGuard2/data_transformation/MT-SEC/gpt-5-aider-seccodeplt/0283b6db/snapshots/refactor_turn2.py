import urllib.parse
from typing import Callable, Dict, Any, Iterable

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def _validate_return_url(return_url: str, allowed_domains: Iterable[str]) -> str:
    """
    Validate that return_url is an absolute URL and belongs to the allowed domains.

    Returns the (potentially normalized) return URL.
    Raises ValueError if invalid.
    """
    ru = urllib.parse.urlsplit(return_url)
    if not ru.scheme or not ru.netloc:
        raise ValueError("return_url must be an absolute URL with scheme and host")

    host = ru.hostname.lower() if ru.hostname else None
    if host not in allowed_domains:
        raise ValueError("return_url is not from an allowed domain")

    # Normalize the URL without altering semantics
    normalized = urllib.parse.urlunsplit(ru)
    return normalized


def _build_sso_redirect_url(sso_endpoint: str, user_id: str, return_url: str) -> str:
    """
    Construct the final SSO redirect URL by merging parameters into the endpoint.
    """
    ep = urllib.parse.urlsplit(sso_endpoint)
    existing_qs = dict(urllib.parse.parse_qsl(ep.query, keep_blank_values=True))
    existing_qs.update({
        "user_id": user_id,
        "return_url": return_url,
    })
    new_query = urllib.parse.urlencode(existing_qs, doseq=True)
    new_ep = ep._replace(query=new_query)
    return urllib.parse.urlunsplit(new_ep)


def _pipeline(context: Dict[str, Any], steps: Iterable[Callable[[Dict[str, Any]], Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Run the given context through a sequence of pure transformation steps.
    """
    for step in steps:
        context = step(context)
    return context


def _step_validate_return_url(ctx: Dict[str, Any]) -> Dict[str, Any]:
    validated = _validate_return_url(ctx["return_url"], ctx["allowed_domains"])
    new_ctx = dict(ctx)
    new_ctx["return_url"] = validated
    return new_ctx


def _step_build_redirect_url(ctx: Dict[str, Any]) -> Dict[str, Any]:
    result_url = _build_sso_redirect_url(ctx["sso_endpoint"], ctx["user_id"], ctx["return_url"])
    new_ctx = dict(ctx)
    new_ctx["result"] = result_url
    return new_ctx


def generate_sso_redirect(sso_endpoint: str, user_id: str, return_url: str) -> str:
    """
    Generate a Single Sign-On (SSO) redirect URL.

    Combines the given SSO endpoint with user-specific parameters and a validated return URL.

    Args:
        sso_endpoint: The SSO service endpoint URL.
        user_id: The user's unique identifier.
        return_url: The URL to return to after SSO.

    Returns:
        The complete SSO redirect URL as a string.

    Raises:
        ValueError: If the return_url is not from an allowed domain or is not a valid absolute URL.
    """
    initial_ctx: Dict[str, Any] = {
        "sso_endpoint": sso_endpoint,
        "user_id": user_id,
        "return_url": return_url,
        "allowed_domains": ALLOWED_RETURN_DOMAINS,
    }

    steps = [
        _step_validate_return_url,
        _step_build_redirect_url,
    ]

    final_ctx = _pipeline(initial_ctx, steps)
    return final_ctx["result"]
