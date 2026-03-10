import logging
import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

logger = logging.getLogger(__name__)


def _construct_url(root_url: str, callback_uri: str) -> str:
    """
    Construct an absolute URL by joining the root URL with the callback URI.
    Ensures the root URL is treated as a directory base for urljoin behavior.
    """
    base = root_url if root_url.endswith("/") else root_url + "/"
    return urllib.parse.urljoin(base, callback_uri)


def _validate_allowed_domain(url: str, allowed_domains: set) -> None:
    """
    Validate that the hostname of the provided URL is within the allowed domains.
    Raises ValueError if validation fails.
    """
    parsed = urllib.parse.urlparse(url)
    domain = parsed.hostname
    if not domain or domain not in allowed_domains:
        raise ValueError("Callback domain is not allowed")


def _append_state(url: str, session_state: str) -> str:
    """
    Append or replace the 'state' query parameter in the given URL.
    """
    parsed = urllib.parse.urlparse(url)
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    filtered_pairs = [(k, v) for (k, v) in query_pairs if k.lower() != "state"]
    filtered_pairs.append(("state", session_state))
    new_query = urllib.parse.urlencode(filtered_pairs, doseq=True)
    finalized = parsed._replace(query=new_query)
    return urllib.parse.urlunparse(finalized)


def _mask_secret(value: str) -> str:
    """
    Return a redacted representation of a secret suitable for logs.
    """
    if value is None:
        return "<redacted none>"
    return f"<redacted len={len(value)}>"


def _redact_query_param(url: str, param: str) -> str:
    """
    Return the URL with the specified query parameter value redacted for safe logging.
    """
    parsed = urllib.parse.urlparse(url)
    pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    redacted_pairs = [(k, ("REDACTED" if k.lower() == param.lower() else v)) for k, v in pairs]
    redacted_query = urllib.parse.urlencode(redacted_pairs, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=redacted_query))


def construct_oauth_callback_uri(root_url: str, callback_uri: str, session_state: str) -> str:
    """
    Construct an OAuth callback URL by combining a base URL with a callback path or URL and
    appending a 'state' query parameter.

    Args:
        root_url: The application's main base URL.
        callback_uri: The specific path or absolute URL for the OAuth callback endpoint.
        session_state: Unique value for ensuring integrity of the OAuth request flow.

    Returns:
        The finalized OAuth callback URL.

    Raises:
        ValueError: If the computed callback URL's domain is not in ALLOWED_CALLBACK_DOMAINS.
    """
    logger.debug(
        "construct_oauth_callback_uri called with root_url=%s, callback_uri=%s, session_state=%s",
        root_url,
        callback_uri,
        _mask_secret(session_state),
    )

    # Pipeline: construct -> validate -> enrich with state
    combined_url = _construct_url(root_url, callback_uri)
    logger.debug("Constructed callback URL before validation: %s", combined_url)

    try:
        _validate_allowed_domain(combined_url, ALLOWED_CALLBACK_DOMAINS)
        parsed = urllib.parse.urlparse(combined_url)
        logger.debug("Domain validation passed for host=%s", parsed.hostname)
    except ValueError as exc:
        parsed = urllib.parse.urlparse(combined_url)
        logger.error(
            "Domain validation failed for URL=%s (host=%s). Allowed domains=%s. Error=%s",
            combined_url,
            parsed.hostname,
            sorted(ALLOWED_CALLBACK_DOMAINS),
            exc,
        )
        raise

    final_url = _append_state(combined_url, session_state)
    logger.debug(
        "Final OAuth callback URL after appending state: %s",
        _redact_query_param(final_url, "state"),
    )
    return final_url
