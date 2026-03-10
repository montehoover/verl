"""
Utilities to construct and validate OAuth callback URLs.

This module provides a main function `create_oauth_callback_url` that builds a
complete OAuth callback URL from a base URL, a callback path, and a state
parameter. It includes helper functions to normalize the base URL, join and
parse the components, validate the resulting domain against an allowlist, and
append the state parameter to the query string.

Logging is incorporated to aid debugging and maintenance by recording the
inputs, intermediate URL construction, final (redacted) callback URL, and any
exceptions encountered during validation.
"""

import logging
import urllib.parse
from typing import Optional, Set

ALLOWED_CALLBACK_DOMAINS = {
    'auth.myapp.com',
    'login.myapp.org',
    'oauth.myapp.net',
}

logger = logging.getLogger(__name__)


def _normalize_base_url(base_url: str) -> str:
    """
    Ensure the provided base URL includes a scheme.

    If the input lacks a scheme (e.g., 'auth.myapp.com'), this function
    defaults to using HTTPS by prefixing 'https://'.

    Args:
        base_url: The base URL string to normalize.

    Returns:
        A URL string guaranteed to have a scheme (e.g., 'https://...').
    """
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme:
        return base_url

    normalized = f'https://{base_url}'
    logger.debug("Normalized base URL: '%s' -> '%s'", base_url, normalized)
    return normalized


def _join_and_parse_url(base_url: str, callback_path: str) -> urllib.parse.ParseResult:
    """
    Combine the base URL and callback path, returning a parsed URL.

    This function uses urllib.parse.urljoin to correctly handle path joining,
    accounting for leading/trailing slashes, and then parses the result.

    Args:
        base_url: The normalized base URL including a scheme.
        callback_path: The callback path to append to the base URL.

    Returns:
        A urllib.parse.ParseResult for the combined URL.
    """
    joined_url = urllib.parse.urljoin(base_url, callback_path)
    logger.debug("Joined base and path into URL: %s", joined_url)
    return urllib.parse.urlparse(joined_url)


def _ensure_allowed_domain(
    parsed_url: urllib.parse.ParseResult,
    allowed_domains: Set[str],
) -> None:
    """
    Validate that the parsed URL's hostname is in the allowlist.

    Args:
        parsed_url: The parsed URL to validate.
        allowed_domains: A set of allowed domain names.

    Raises:
        ValueError: If the hostname is missing or not in the allowlist.
    """
    hostname = (parsed_url.hostname or '').lower()
    if not hostname or hostname not in allowed_domains:
        logger.error(
            "Disallowed callback domain: hostname='%s', allowed=%s",
            hostname,
            sorted(allowed_domains),
        )
        raise ValueError('Callback URL domain is not allowed')
    logger.debug("Allowed callback domain validated: %s", hostname)


def _with_state_query(
    parsed_url: urllib.parse.ParseResult,
    state: str,
) -> urllib.parse.ParseResult:
    """
    Return a new parsed URL with the provided 'state' query parameter applied.

    If a 'state' parameter already exists, it will be overwritten to ensure the
    provided value is used. Existing query parameters are preserved.

    Args:
        parsed_url: The parsed URL to modify.
        state: The state value to include in the query string.

    Returns:
        A new urllib.parse.ParseResult with an updated query string.
    """
    query_pairs = urllib.parse.parse_qsl(
        parsed_url.query,
        keep_blank_values=True,
    )
    query_dict = dict(query_pairs)
    query_dict['state'] = state
    new_query = urllib.parse.urlencode(query_dict, doseq=True)
    updated = parsed_url._replace(query=new_query)
    logger.debug("Applied state parameter to URL path: %s", updated.path)
    return updated


def _unparse_url(parsed_url: urllib.parse.ParseResult) -> str:
    """
    Convert a parsed URL back to its string representation.

    Args:
        parsed_url: The parsed URL to convert.

    Returns:
        The full URL as a string.
    """
    return urllib.parse.urlunparse(parsed_url)


def _redact_query_params_in_url(
    url: str,
    keys: Optional[Set[str]] = None,
) -> str:
    """
    Redact sensitive query parameter values in a URL string for safe logging.

    Args:
        url: The URL string whose query parameters may contain sensitive data.
        keys: A set of query parameter names to redact. If None, redacts
            {'state'} by default.

    Returns:
        A URL string with specified query parameters' values replaced
        with 'REDACTED'.
    """
    redact_keys = keys or {'state'}
    parsed = urllib.parse.urlparse(url)
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    redacted_pairs = [
        (k, 'REDACTED' if k in redact_keys else v) for k, v in query_pairs
    ]
    redacted_query = urllib.parse.urlencode(redacted_pairs, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=redacted_query))


def create_oauth_callback_url(
    base_url: str,
    callback_path: str,
    state: str,
) -> str:
    """
    Create a complete OAuth callback URL by combining the base URL with a
    callback path and appending a 'state' query parameter.

    Args:
        base_url: The base URL of the application (e.g., 'https://auth.myapp.com').
        callback_path: The path for the OAuth callback (e.g., '/oauth/callback').
        state: A unique state parameter for the OAuth flow.

    Returns:
        The complete OAuth callback URL as a string.

    Raises:
        ValueError: If the resulting callback URL is not from an allowed domain.
    """
    logger.info(
        "Creating OAuth callback URL: base_url=%s, callback_path=%s",
        base_url,
        callback_path,
    )
    try:
        normalized_base = _normalize_base_url(base_url)
        parsed = _join_and_parse_url(normalized_base, callback_path)
        _ensure_allowed_domain(parsed, ALLOWED_CALLBACK_DOMAINS)
        parsed_with_state = _with_state_query(parsed, state)
        final_url = _unparse_url(parsed_with_state)
        logger.info(
            "Constructed OAuth callback URL: %s",
            _redact_query_params_in_url(final_url, {'state'}),
        )
        return final_url
    except ValueError:
        logger.exception(
            "Failed to create OAuth callback URL for base_url=%s, path=%s",
            base_url,
            callback_path,
        )
        raise
