import urllib.parse
import logging
import hashlib

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

logger = logging.getLogger(__name__)


def _construct_callback_url(host_url: str, path_callback: str) -> str:
    """
    Pure function: Construct the callback URL from a host URL and a callback path/URL.
    """
    return urllib.parse.urljoin(host_url, path_callback)


def _validate_callback_domain(host_url: str, callback_url: str) -> str:
    """
    Pure function: Validate that the callback URL's domain is either the same as the host
    domain or present in ALLOWED_CALLBACK_DOMAINS. Returns the unmodified callback_url
    on success for easy pipeline chaining.
    """
    host_hostname = urllib.parse.urlparse(host_url).hostname
    cb_hostname = urllib.parse.urlparse(callback_url).hostname

    if cb_hostname is None:
        raise ValueError("Callback URL must include a valid domain")

    if not (cb_hostname == host_hostname or cb_hostname in ALLOWED_CALLBACK_DOMAINS):
        raise ValueError("Callback URL domain is not permitted")

    return callback_url


def _upsert_state_param(callback_url: str, session_id: str) -> str:
    """
    Pure function: Ensure the 'state' query parameter is set to session_id.
    If present, it is replaced; otherwise, it is added.
    """
    cb_parsed = urllib.parse.urlparse(callback_url)
    existing_params = urllib.parse.parse_qsl(cb_parsed.query, keep_blank_values=True)
    filtered_params = [(k, v) for (k, v) in existing_params if k.lower() != 'state']
    filtered_params.append(('state', session_id))
    new_query = urllib.parse.urlencode(filtered_params)
    return urllib.parse.urlunparse(cb_parsed._replace(query=new_query))


def _session_id_log_repr(session_id: str) -> str:
    """
    Produce a safe representation of the session_id for logs (non-reversible).
    """
    digest = hashlib.sha256(session_id.encode('utf-8')).hexdigest()[:12]
    return f"sha256:{digest}"


def _sanitize_state_in_url(url: str) -> str:
    """
    Return a version of the URL where any 'state' parameter value is redacted.
    """
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    sanitized = [(k, ('[redacted]' if k.lower() == 'state' else v)) for k, v in params]
    new_query = urllib.parse.urlencode(sanitized)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def generate_oauth_url(host_url: str, path_callback: str, session_id: str) -> str:
    """
    Construct an OAuth callback URL by combining the base host URL with the callback path
    and appending a state parameter derived from session_id. Enforces that the resulting
    callback URL domain is either the same as the host URL's domain or included in
    ALLOWED_CALLBACK_DOMAINS.

    Args:
        host_url: Root URL for the application (e.g., "https://app.myapp.com").
        path_callback: Callback handler endpoint path or absolute URL.
        session_id: Unique identifier used as the OAuth state parameter.

    Returns:
        The fully assembled OAuth callback URL as a string.

    Raises:
        ValueError: If the callback URL domain is neither the same as the host domain
                    nor included in ALLOWED_CALLBACK_DOMAINS.
    """
    session_repr = _session_id_log_repr(session_id)
    logger.debug(
        "generate_oauth_url inputs: host_url=%r path_callback=%r session_id=%s",
        host_url,
        path_callback,
        session_repr,
    )

    callback_url = _construct_callback_url(host_url, path_callback)
    logger.debug("Constructed callback URL: %s", callback_url)

    try:
        validated_url = _validate_callback_domain(host_url, callback_url)
    except ValueError as e:
        logger.error(
            "OAuth callback domain validation failed: host_url=%s callback_url=%s error=%s",
            host_url,
            callback_url,
            str(e),
        )
        raise

    host_hostname = urllib.parse.urlparse(host_url).hostname
    cb_hostname = urllib.parse.urlparse(validated_url).hostname
    logger.debug(
        "Callback domain validated: host_domain=%s callback_domain=%s",
        host_hostname,
        cb_hostname,
    )

    final_url = _upsert_state_param(validated_url, session_id)
    redacted_final = _sanitize_state_in_url(final_url)
    logger.info("Generated OAuth URL (state redacted): %s", redacted_final)

    return final_url
