import urllib.parse
import logging

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

logger = logging.getLogger(__name__)


def validate_and_normalize_url_parts(
    base_api_url: str,
    api_endpoint: str,
    approved_domains: set[str],
    allowed_endpoints: set[str],
) -> tuple[urllib.parse.ParseResult, str]:
    """
    Pure function to validate and normalize the base API URL and endpoint.

    Returns:
        A tuple of (parsed_base_url, normalized_endpoint_path)

    Raises:
        ValueError: If inputs are invalid, domain not approved, or endpoint not allowed.
    """
    # Validate and normalize base_api_url
    if not isinstance(base_api_url, str) or not base_api_url.strip():
        raise ValueError("Base API URL must be a non-empty string")

    parsed_base = urllib.parse.urlparse(base_api_url.strip())
    if not parsed_base.netloc:
        candidate = base_api_url.strip()
        if candidate.startswith("//"):
            candidate = "https:" + candidate
        elif not candidate.startswith(("http://", "https://")):
            candidate = "https://" + candidate
        parsed_base = urllib.parse.urlparse(candidate)

    hostname = parsed_base.hostname
    if not hostname or hostname not in approved_domains:
        raise ValueError("Base API domain is not approved")

    # Validate and normalize api_endpoint
    if not isinstance(api_endpoint, str) or not api_endpoint.strip():
        raise ValueError("API endpoint must be a non-empty string")

    endpoint_parsed = urllib.parse.urlparse(api_endpoint.strip())
    endpoint_path = endpoint_parsed.path or ""
    if not endpoint_path.startswith("/"):
        endpoint_path = "/" + endpoint_path

    while "//" in endpoint_path:
        endpoint_path = endpoint_path.replace("//", "/")

    if endpoint_path not in allowed_endpoints:
        raise ValueError("Endpoint is not allowed")

    return parsed_base, endpoint_path


def build_query_string(query_options: dict | None) -> str:
    """
    Pure function to construct a query string from the provided options.

    Returns:
        A URL-encoded query string (without the leading '?').
    """
    if query_options is None:
        return ""
    if not isinstance(query_options, dict):
        raise ValueError("query_options must be a dictionary if provided")

    cleaned = {k: v for k, v in query_options.items() if v is not None}
    return urllib.parse.urlencode(cleaned, doseq=True)


def construct_api_redirect(base_api_url: str, api_endpoint: str, query_options: dict | None = None) -> str:
    """
    Construct a full API redirect URL from a base API URL, an endpoint, and optional query params.

    Args:
        base_api_url: Base address of an API (e.g., 'https://api.myservice.com' or 'api.myservice.com').
        api_endpoint: Endpoint path (e.g., '/v1/data').
        query_options: Optional dict of query parameters; values can be scalars or sequences.

    Returns:
        A fully constructed API redirect URL as a string.

    Raises:
        ValueError: If the domain is not approved or the endpoint is not allowed.
    """
    logger.info(
        "API redirect construction attempt: base_api_url=%r, api_endpoint=%r, query_options=%r",
        base_api_url,
        api_endpoint,
        query_options,
    )

    try:
        parsed_base, endpoint_path = validate_and_normalize_url_parts(
            base_api_url,
            api_endpoint,
            APPROVED_API_DOMAINS,
            set(ALLOWED_ENDPOINTS),
        )

        query_str = build_query_string(query_options)

        final_url = urllib.parse.urlunparse((
            parsed_base.scheme or "https",
            parsed_base.netloc,
            endpoint_path,
            "",
            query_str,
            "",
        ))

        # Final safety check after construction.
        final_parsed = urllib.parse.urlparse(final_url)
        if final_parsed.hostname not in APPROVED_API_DOMAINS:
            raise ValueError("Generated URL domain is not allowed")
        if final_parsed.path not in ALLOWED_ENDPOINTS:
            raise ValueError("Generated URL endpoint is not allowed")

        logger.info("API redirect constructed: final_url=%s", final_url)
        return final_url
    except ValueError:
        logger.exception(
            "API redirect construction failed: base_api_url=%r, api_endpoint=%r, query_options=%r",
            base_api_url,
            api_endpoint,
            query_options,
        )
        raise
