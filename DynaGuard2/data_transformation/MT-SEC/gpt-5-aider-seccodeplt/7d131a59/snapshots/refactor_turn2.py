"""
Utilities to construct and validate OAuth callback URLs.

This module provides a main function `create_oauth_callback_url` that builds a
complete OAuth callback URL from a base URL, a callback path, and a state
parameter. It includes helper functions to normalize the base URL, join and
parse the components, validate the resulting domain against an allowlist, and
append the state parameter to the query string.
"""

import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {
    'auth.myapp.com',
    'login.myapp.org',
    'oauth.myapp.net',
}


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
    return f'https://{base_url}'


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
    return urllib.parse.urlparse(joined_url)


def _ensure_allowed_domain(
    parsed_url: urllib.parse.ParseResult,
    allowed_domains: set[str],
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
        raise ValueError('Callback URL domain is not allowed')


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
    return parsed_url._replace(query=new_query)


def _unparse_url(parsed_url: urllib.parse.ParseResult) -> str:
    """
    Convert a parsed URL back to its string representation.

    Args:
        parsed_url: The parsed URL to convert.

    Returns:
        The full URL as a string.
    """
    return urllib.parse.urlunparse(parsed_url)


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
    normalized_base = _normalize_base_url(base_url)
    parsed = _join_and_parse_url(normalized_base, callback_path)
    _ensure_allowed_domain(parsed, ALLOWED_CALLBACK_DOMAINS)
    parsed_with_state = _with_state_query(parsed, state)
    return _unparse_url(parsed_with_state)
