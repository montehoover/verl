"""
Utilities for constructing validated API redirect URLs.

This module provides a high-level function `create_redirect_url` along with
supporting pure helper functions to parse and validate the base URL, normalize
and validate endpoints, build query strings, and construct the final URL.

Logging
-------
This module uses the standard library `logging` package. To observe debug logs
from this module, configure logging in your application, for example:

    import logging

    logging.basicConfig(level=logging.DEBUG)
"""

import logging
import urllib.parse

APPROVED_API_DOMAINS = {
    'api.myservice.com',
    'api-test.myservice.com',
    'api-staging.myservice.com',
}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

logger = logging.getLogger(__name__)


def _parse_and_validate_base_url(base_url_api: str) -> urllib.parse.ParseResult:
    """
    Parse and validate the base API URL.

    Ensures:
      - The URL contains a scheme and network location.
      - The hostname is present and within APPROVED_API_DOMAINS.

    Args:
        base_url_api: The base API URL (e.g., "https://api.myservice.com").

    Returns:
        The parsed URL as a urllib.parse.ParseResult.

    Raises:
        ValueError: If the URL is missing a scheme/host or the domain is not
            approved.
    """
    parsed = urllib.parse.urlparse(base_url_api)
    logger.debug(
        "Parsing base URL: raw=%r -> scheme=%r, netloc=%r, path=%r",
        base_url_api, parsed.scheme, parsed.netloc, parsed.path,
    )

    if not parsed.scheme or not parsed.netloc:
        logger.error(
            "Invalid base_url_api: missing scheme or host: %r", base_url_api
        )
        raise ValueError(
            "base_url_api must include a scheme and host "
            "(e.g., https://api.myservice.com)"
        )

    hostname = parsed.hostname
    logger.debug("Extracted hostname: %r", hostname)

    if hostname not in APPROVED_API_DOMAINS:
        logger.error("Unapproved domain in base_url_api: %r", hostname)
        raise ValueError("Base URL domain is not approved")

    return parsed


def _normalize_and_validate_endpoint(desired_endpoint: str) -> str:
    """
    Normalize and validate the API endpoint.

    Normalization:
      - Ensures a single leading slash.
      - Removes a trailing slash for non-root paths.

    Validation:
      - Ensures the normalized endpoint is present in ALLOWED_ENDPOINTS.

    Args:
        desired_endpoint: The endpoint to normalize and validate.

    Returns:
        The normalized endpoint string.

    Raises:
        ValueError: If the endpoint is not in ALLOWED_ENDPOINTS.
    """
    normalized = "/" + desired_endpoint.lstrip("/")
    if len(normalized) > 1:
        normalized = normalized.rstrip("/")

    logger.debug(
        "Normalized endpoint: input=%r -> normalized=%r",
        desired_endpoint, normalized
    )

    if normalized not in ALLOWED_ENDPOINTS:
        logger.error("Endpoint is not allowed: %r", normalized)
        raise ValueError("Endpoint is not allowed")

    return normalized


def _build_query(optional_params: dict | None) -> str:
    """
    Build a URL-encoded query string from a dictionary of parameters.

    Uses urllib.parse.urlencode with doseq=True to support multiple values per
    key when lists/tuples are provided.

    Args:
        optional_params: A dictionary of query parameters, or None.

    Returns:
        A URL-encoded query string without a leading '?' (empty string if no
        parameters).
    """
    if not optional_params:
        logger.debug("No query parameters provided; query string will be empty.")
        return ""

    query = urllib.parse.urlencode(optional_params, doseq=True, safe="")
    logger.debug("Built query string from parameters %r -> %r", optional_params, query)
    return query


def _construct_url(
    parsed_base: urllib.parse.ParseResult,
    endpoint: str,
    query: str,
) -> str:
    """
    Construct the final URL from validated components.

    Args:
        parsed_base: The parsed base URL.
        endpoint: The validated and normalized endpoint path.
        query: The URL-encoded query string (no leading '?').

    Returns:
        The fully constructed URL as a string.
    """
    final_url = urllib.parse.urlunparse(
        (
            parsed_base.scheme,
            parsed_base.netloc,
            endpoint,
            "",   # params (deprecated)
            query,
            "",   # fragment
        )
    )
    logger.debug(
        "Constructed URL: scheme=%r, netloc=%r, path=%r, query=%r -> %s",
        parsed_base.scheme, parsed_base.netloc, endpoint, query, final_url
    )
    return final_url


def _final_safety_check(url: str) -> None:
    """
    Perform a final verification to ensure the URL conforms to approved domain
    and endpoint.

    Args:
        url: The URL to validate.

    Raises:
        ValueError: If the URL's hostname is not approved or its path is not
            within allowed endpoints.
    """
    parsed = urllib.parse.urlparse(url)
    logger.debug(
        "Final safety check: hostname=%r, path=%r", parsed.hostname, parsed.path
    )

    if parsed.hostname not in APPROVED_API_DOMAINS or parsed.path not in ALLOWED_ENDPOINTS:
        logger.error(
            "Generated URL diverges from approved domain or endpoint: %s",
            url,
        )
        raise ValueError("Generated URL diverges from approved domain or endpoint")


def create_redirect_url(
    base_url_api: str,
    desired_endpoint: str,
    optional_params: dict | None = None,
) -> str:
    """
    Create a full redirect URL for API requests.

    The function combines the base API URL, a specific endpoint, and optional
    query parameters to produce a complete redirect URL. It performs input
    validation and enforces domain and endpoint allowlists.

    Args:
        base_url_api: String representing the main API base URL.
        desired_endpoint: String representing the desired API endpoint.
        optional_params: An optional dictionary for query string parameters.

    Returns:
        A complete API redirect URL as a string.

    Raises:
        ValueError: If inputs are invalid or if the generated URL diverges from
            an approved domain or endpoint.
    """
    logger.debug(
        "create_redirect_url called with base_url_api=%r, desired_endpoint=%r, "
        "optional_params=%r",
        base_url_api, desired_endpoint, optional_params
    )

    if not isinstance(base_url_api, str) or not base_url_api.strip():
        logger.error("base_url_api must be a non-empty string: %r", base_url_api)
        raise ValueError("base_url_api must be a non-empty string")

    if not isinstance(desired_endpoint, str) or not desired_endpoint.strip():
        logger.error(
            "desired_endpoint must be a non-empty string: %r", desired_endpoint
        )
        raise ValueError("desired_endpoint must be a non-empty string")

    if optional_params is not None and not isinstance(optional_params, dict):
        logger.error(
            "optional_params must be a dictionary if provided: %r",
            optional_params,
        )
        raise ValueError("optional_params must be a dictionary if provided")

    parsed_base = _parse_and_validate_base_url(base_url_api)
    normalized_endpoint = _normalize_and_validate_endpoint(desired_endpoint)
    query = _build_query(optional_params)
    final_url = _construct_url(parsed_base, normalized_endpoint, query)
    _final_safety_check(final_url)

    logger.debug("Successfully created redirect URL: %s", final_url)
    return final_url
