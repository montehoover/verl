"""
Utilities to construct and validate API redirect URLs.

This module exposes a main function, `generate_api_redirect`, which builds a
fully-qualified URL from a base API URL, an endpoint path, and optional query
parameters. The function enforces that only approved API domains and allowed
endpoints are used, mitigating the risk of open redirects or requests to
unexpected destinations.
"""

import urllib.parse

# Approved base API domains (hostnames only; ports are ignored during validation).
APPROVED_API_DOMAINS = {
    'api.myservice.com',
    'api-test.myservice.com',
    'api-staging.myservice.com',
}

# Allowed API endpoints (exact path matches after normalization).
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _normalize_and_parse_base_url(base_url: str) -> urllib.parse.ParseResult:
    """
    Normalize and parse the provided base URL.

    Ensures that the URL has a scheme. If missing, defaults to "https://".
    Returns a ParseResult suitable for further validation and reconstruction.

    Args:
        base_url: The base URL string (e.g., "https://api.myservice.com").

    Returns:
        urllib.parse.ParseResult: The parsed and normalized result.
    """
    # Attempt to parse the provided base URL as-is.
    parsed_base = urllib.parse.urlparse(base_url)

    # If no scheme is present, assume HTTPS for security and parse again.
    if not parsed_base.scheme:
        parsed_base = urllib.parse.urlparse("https://" + base_url)

    return parsed_base


def _validate_base_domain(parsed_base: urllib.parse.ParseResult) -> None:
    """
    Validate that the parsed base URL points to an approved domain.

    Validation is done against the hostname only (ports are ignored).

    Args:
        parsed_base: A ParseResult generated from the base URL.

    Raises:
        ValueError: If the hostname is missing or not approved.
    """
    # Hostname is required; urlparse provides it separate from netloc.
    if not parsed_base.hostname:
        raise ValueError("base_url must include a valid hostname")

    # Ensure the hostname matches one of the approved API domains.
    if parsed_base.hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Base URL domain is not approved")


def _parse_validate_normalize_api_path(api_path: str) -> str:
    """
    Parse, validate, and normalize the API path.

    Rejects values that contain a scheme, netloc, query, params, or fragment,
    ensuring the input is strictly a path. The result is normalized to:
      - Always start with a single leading slash.
      - Have no trailing slash (unless it's the root "/").

    Args:
        api_path: The path component of the API endpoint (e.g., "/v1/data").

    Returns:
        str: The normalized path.

    Raises:
        ValueError: If the path is not a pure path or is not in allowed endpoints.
    """
    parsed_path = urllib.parse.urlparse(api_path)

    # The endpoint must be a pure path without any URL parts other than path.
    if (
        parsed_path.scheme
        or parsed_path.netloc
        or parsed_path.query
        or parsed_path.params
        or parsed_path.fragment
    ):
        raise ValueError(
            "api_path must be a path without scheme, domain, query, "
            "parameters, or fragment"
        )

    # Ensure a single leading slash and remove trailing slash for consistency.
    normalized_path = "/" + (parsed_path.path or "").lstrip("/")
    if len(normalized_path) > 1:
        normalized_path = normalized_path.rstrip("/")

    # Only allow explicit, exact matches to the approved endpoint list.
    if normalized_path not in ALLOWED_ENDPOINTS:
        raise ValueError("Endpoint is not allowed")

    return normalized_path


def _build_query(params: dict | None) -> str:
    """
    Build a URL-encoded query string from a parameters dictionary.

    - Keys are coerced to strings.
    - Values of None are skipped.
    - Lists or tuples are supported via doseq=True.

    Args:
        params: Optional dictionary of query parameters.

    Returns:
        str: The URL-encoded query string (without a leading '?'), or an empty
        string if no parameters were provided.
    """
    if not params:
        return ""

    # Filter out None values and coerce keys to strings.
    normalized_params = {}
    for key, value in params.items():
        if value is None:
            continue
        normalized_params[str(key)] = value

    # doseq=True expands list-like values into repeated query keys.
    return urllib.parse.urlencode(normalized_params, doseq=True)


def _construct_url(
    parsed_base: urllib.parse.ParseResult,
    path: str,
    query: str,
) -> str:
    """
    Construct the final URL from the parsed base, normalized path, and query.

    Args:
        parsed_base: ParseResult for the base URL (scheme and netloc will be used).
        path: Normalized path (must begin with '/').
        query: URL-encoded query string without a leading '?'.

    Returns:
        str: The constructed URL string.
    """
    return urllib.parse.urlunparse(
        (
            parsed_base.scheme,
            parsed_base.netloc,
            path,
            "",      # params (deprecated; unused)
            query,   # query string without the leading '?'
            "",      # fragment
        )
    )


def generate_api_redirect(
    base_url: str,
    api_path: str,
    params: dict | None = None,
) -> str:
    """
    Generate a validated API redirect URL.

    Combines a validated base URL, a normalized/validated API path, and an
    optional set of query parameters into a single, fully-qualified URL.

    Args:
        base_url: Base address of the API (e.g., "https://api.myservice.com").
        api_path: Endpoint path (e.g., "/v1/data").
        params: Optional dictionary of query parameters.

    Returns:
        str: Fully constructed API redirect URL.

    Raises:
        ValueError: If the base URL is not in an approved domain or the
        endpoint path is not in the list of allowed endpoints, or if inputs
        are malformed.

    Examples:
        >>> generate_api_redirect("api.myservice.com", "/v1/data", {"q": "test"})
        'https://api.myservice.com/v1/data?q=test'
    """
    # Basic input validation for required types and non-empty strings.
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")

    if not isinstance(api_path, str) or not api_path.strip():
        raise ValueError("api_path must be a non-empty string")

    if params is not None and not isinstance(params, dict):
        raise ValueError("params must be a dictionary if provided")

    # Normalize and parse the base URL, then ensure it belongs to an approved domain.
    parsed_base = _normalize_and_parse_base_url(base_url)
    _validate_base_domain(parsed_base)

    # Normalize and validate the API path against the allowlist.
    normalized_path = _parse_validate_normalize_api_path(api_path)

    # Build the query string from provided parameters.
    query = _build_query(params)

    # Construct and return the final URL.
    final_url = _construct_url(parsed_base, normalized_path, query)
    return final_url
