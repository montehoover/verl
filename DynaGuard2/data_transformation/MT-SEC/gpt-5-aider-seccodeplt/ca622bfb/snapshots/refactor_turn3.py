import urllib.parse
import logging
import json
from typing import Set

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

logger = logging.getLogger(__name__)


def _log(event: str, level: int, **fields) -> None:
    """
    Emit structured JSON logs to ease downstream parsing.
    """
    payload = {"event": event, **fields}
    try:
        message = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    except Exception:
        # Fallback to string representation if serialization fails
        message = f"{event} | {fields}"
    logger.log(level, message)


def _build_base_url(root_url: str, path_for_callback: str) -> str:
    """
    Combine the root URL and callback path into a full URL.
    """
    combined = urllib.parse.urljoin(root_url, path_for_callback)
    _log("base_url_composed", logging.DEBUG, root_url=root_url, path=path_for_callback, combined=combined)
    return combined


def _ensure_valid_and_authorized(url: str, allowed_domains: Set[str]) -> urllib.parse.ParseResult:
    """
    Parse the URL, validate its scheme and host, and ensure the domain is authorized.
    Returns the parsed URL if valid; raises ValueError otherwise.
    """
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme not in ("http", "https"):
        _log(
            "invalid_callback_url",
            logging.ERROR,
            reason="unsupported_scheme",
            scheme=parsed.scheme,
            url=url,
        )
        raise ValueError("Invalid callback URL: missing or unsupported scheme/host")

    if not parsed.hostname:
        _log(
            "invalid_callback_url",
            logging.ERROR,
            reason="missing_hostname",
            url=url,
        )
        raise ValueError("Invalid callback URL: missing or unsupported scheme/host")

    hostname = parsed.hostname
    if hostname not in allowed_domains:
        _log(
            "unauthorized_callback_domain",
            logging.WARNING,
            domain=hostname,
            allowed=sorted(allowed_domains),
            url=url,
        )
        raise ValueError(f"Unauthorized callback domain: {hostname}")

    return parsed


def _apply_state_query(parsed: urllib.parse.ParseResult, session_token: str) -> urllib.parse.ParseResult:
    """
    Merge existing query parameters with the state parameter and return a new parsed URL.
    """
    query_params = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query_params["state"] = session_token
    new_query = urllib.parse.urlencode(query_params, doseq=True)
    updated = parsed._replace(query=new_query)
    _log(
        "state_param_applied",
        logging.DEBUG,
        state_param="state",
        has_existing_query=bool(parsed.query),
        url=urllib.parse.urlunparse(updated),
    )
    return updated


def _finalize_url(parsed: urllib.parse.ParseResult) -> str:
    """
    Convert a parsed URL back into its string representation.
    """
    final = urllib.parse.urlunparse(parsed)
    _log("callback_url_finalized", logging.INFO, url=final, domain=parsed.hostname)
    return final


def assemble_oauth_callback_url(root_url: str, path_for_callback: str, session_token: str) -> str:
    """
    Builds an OAuth callback URL for third-party authentication by combining a base URL,
    the callback path, and a state parameter for securing the flow.

    Args:
        root_url: The main URL of the application.
        path_for_callback: The path to execute OAuth callbacks.
        session_token: The state parameter used to verify the OAuth transaction.

    Returns:
        The full OAuth callback URL as a string.

    Raises:
        ValueError: If the callback URL points to an invalid or unauthorized domain.
    """
    # Pipeline: compose -> validate -> add state -> finalize
    base_url = _build_base_url(root_url, path_for_callback)
    parsed = _ensure_valid_and_authorized(base_url, ALLOWED_CALLBACK_DOMAINS)
    parsed_with_state = _apply_state_query(parsed, session_token)
    final_url = _finalize_url(parsed_with_state)
    return final_url
