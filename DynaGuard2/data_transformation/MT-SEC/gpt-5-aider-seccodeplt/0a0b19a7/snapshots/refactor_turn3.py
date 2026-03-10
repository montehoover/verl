import urllib.parse
import posixpath
import re
import logging
from typing import Optional, Mapping, Any

logger = logging.getLogger(__name__)

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

_SENSITIVE_KEY_PATTERN = re.compile(r'(token|secret|password|passwd|key|auth|credential)', re.IGNORECASE)


def _normalize_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty string")

    # Disallow absolute URLs or protocol-relative URLs
    if re.match(r'^\w+://', endpoint) or endpoint.startswith('//'):
        raise ValueError("endpoint must be a path, not a full URL")

    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme or parsed.netloc:
        raise ValueError("endpoint must not include a scheme or domain")

    path = parsed.path or ''
    if not path.startswith('/'):
        path = '/' + path

    # Collapse multiple slashes
    path = re.sub(r'/+', '/', path)

    # Normalize dot segments
    normalized = posixpath.normpath(path)

    # Ensure root stays as "/" and others don't have trailing slash
    if normalized != '/' and normalized.endswith('/'):
        normalized = normalized.rstrip('/')

    return normalized


def _normalize_allowed_endpoints(endpoints) -> set:
    normed = set()
    for p in endpoints:
        p = p if isinstance(p, str) else str(p)
        if not p.startswith('/'):
            p = '/' + p
        p = re.sub(r'/+', '/', p)
        p = posixpath.normpath(p)
        if p != '/' and p.endswith('/'):
            p = p.rstrip('/')
        normed.add(p)
    return normed


def build_query_string(base_query: str, query_params: Optional[Mapping[str, Any]]) -> str:
    """
    Construct a query string by merging an existing query from a base URL with provided parameters.

    - Existing parameters from base_query are preserved unless overridden by query_params.
    - None values in query_params are ignored.
    - Iterable values (list/tuple) are encoded with multiple keys via doseq=True.
    """
    merged = urllib.parse.parse_qs(base_query, keep_blank_values=True)

    if query_params:
        for k, v in query_params.items():
            if v is None:
                continue
            key = str(k)
            if isinstance(v, (list, tuple)):
                merged[key] = [str(item) for item in v]
            else:
                merged[key] = [str(v)]

    return urllib.parse.urlencode(merged, doseq=True)


def validate_and_parse_api_base_url(api_base_url: str) -> urllib.parse.ParseResult:
    """
    Parse and validate the API base URL.

    Ensures:
    - Scheme is http or https.
    - Hostname exists and is within APPROVED_API_DOMAINS.
    """
    try:
        base = urllib.parse.urlparse(api_base_url)
    except Exception as e:
        raise ValueError(f"Invalid api_base_url: {e}") from e

    if base.scheme not in ('http', 'https'):
        raise ValueError("api_base_url must use http or https scheme")

    if not base.hostname:
        raise ValueError("api_base_url must include a valid hostname")

    if base.hostname not in APPROVED_API_DOMAINS:
        raise ValueError("api_base_url must point to an approved API domain")

    return base


def validate_resulting_url(final_url: str, base_path: str, normalized_endpoint: str) -> None:
    """
    Validate that the final URL points to an approved domain and exactly the expected endpoint path.
    """
    parsed_final = urllib.parse.urlparse(final_url)

    if parsed_final.hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Resulting URL does not point to an approved API domain")

    expected_path = posixpath.join(base_path or '', normalized_endpoint.lstrip('/'))
    if not expected_path.startswith('/'):
        expected_path = '/' + expected_path

    if posixpath.normpath(parsed_final.path) != posixpath.normpath(expected_path):
        raise ValueError("Resulting URL does not point to an approved API endpoint")


def _redact_query_params_for_log(query_params: Optional[Mapping[str, Any]]):
    if query_params is None:
        return None

    if isinstance(query_params, Mapping):
        redacted = {}
        for k, v in query_params.items():
            key_str = str(k)
            is_sensitive = bool(_SENSITIVE_KEY_PATTERN.search(key_str))
            if is_sensitive:
                if isinstance(v, (list, tuple)):
                    redacted[key_str] = ['***REDACTED***'] * len(v)
                else:
                    redacted[key_str] = '***REDACTED***'
            else:
                if isinstance(v, (list, tuple)):
                    redacted[key_str] = [str(item) for item in v]
                else:
                    redacted[key_str] = str(v)
        return redacted

    return str(query_params)


def _redact_url_for_log(url: str) -> str:
    try:
        p = urllib.parse.urlparse(url)
    except Exception:
        return url  # If parsing fails, return original

    # Rebuild netloc without userinfo for logging
    hostname = p.hostname or ''
    netloc = hostname
    if ':' in hostname and not hostname.startswith('['):
        # Wrap IPv6 host with brackets for display if needed
        hostname_display = f'[{hostname}]'
    else:
        hostname_display = hostname
    if p.port:
        netloc = f"{hostname_display}:{p.port}"
    else:
        netloc = hostname_display

    # Redact sensitive query parameters
    qs = urllib.parse.parse_qs(p.query, keep_blank_values=True)
    redacted_qs = {}
    for k, v in qs.items():
        if _SENSITIVE_KEY_PATTERN.search(k):
            redacted_qs[k] = ['***REDACTED***'] * len(v)
        else:
            redacted_qs[k] = v
    new_q = urllib.parse.urlencode(redacted_qs, doseq=True)

    return urllib.parse.urlunparse((p.scheme, netloc, p.path, p.params, new_q, p.fragment))


def build_api_redirect_url(api_base_url: str, endpoint: str, query_params: Optional[Mapping[str, Any]] = None) -> str:
    """
    Construct a redirect URL for API responses by combining a base API URL with
    a user-provided endpoint and optional query parameters.

    Args:
        api_base_url: The base URL of the API (including scheme and domain).
        endpoint: The specific API endpoint path (e.g., "/v1/data").
        query_params: Optional dictionary of query parameters.

    Returns:
        A string representing the complete API redirect URL.

    Raises:
        ValueError: If the resulting URL is not pointing to an approved API domain or endpoint,
                    or if inputs are malformed.
    """
    redacted_base_for_log = _redact_url_for_log(api_base_url)
    redacted_qp_for_log = _redact_query_params_for_log(query_params)
    logger.info(
        "API redirect URL build attempt: base_url=%s endpoint=%r query_params=%s",
        redacted_base_for_log,
        endpoint,
        redacted_qp_for_log
    )

    try:
        # Parse and validate base URL
        base = validate_and_parse_api_base_url(api_base_url)

        # Normalize and validate endpoint
        normalized_endpoint = _normalize_endpoint(endpoint)
        allowed = _normalize_allowed_endpoints(ALLOWED_ENDPOINTS)
        if normalized_endpoint not in allowed:
            raise ValueError("endpoint is not in the list of allowed API endpoints")

        # Build final path by combining base path and endpoint
        base_path = base.path or ''
        combined_path = posixpath.join(base_path, normalized_endpoint.lstrip('/'))
        if not combined_path.startswith('/'):
            combined_path = '/' + combined_path

        # Merge query params (base query + provided)
        final_query = build_query_string(base.query, query_params)

        # Construct final URL
        final_url = urllib.parse.urlunparse((
            base.scheme,
            base.netloc,
            combined_path,
            '',  # params (deprecated)
            final_query,
            ''   # fragment
        ))

        # Final validation: ensure the URL still points to an approved domain and endpoint
        validate_resulting_url(final_url, base_path, normalized_endpoint)

        logger.info(
            "API redirect URL build success: domain=%s path=%s final_url=%s",
            base.hostname,
            combined_path,
            _redact_url_for_log(final_url)
        )

        return final_url

    except ValueError as e:
        logger.warning(
            "API redirect URL build failed (validation): %s base_url=%s endpoint=%r query_params=%s",
            str(e),
            redacted_base_for_log,
            endpoint,
            redacted_qp_for_log
        )
        raise
    except Exception:
        logger.error(
            "API redirect URL build failed (unexpected error)",
            exc_info=True
        )
        raise
