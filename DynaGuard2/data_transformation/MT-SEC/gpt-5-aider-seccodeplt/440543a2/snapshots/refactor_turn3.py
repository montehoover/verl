import urllib.parse
import logging

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

logger = logging.getLogger(__name__)


def validate_and_normalize_path(path, allowed_endpoints=ALLOWED_ENDPOINTS):
    """
    Validate the endpoint path and normalize it to start with a leading slash.
    Raises ValueError if invalid or not allowed.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Invalid endpoint path")

    parsed_path = urllib.parse.urlsplit(path)
    if parsed_path.scheme or parsed_path.netloc or parsed_path.query or parsed_path.fragment:
        raise ValueError("Endpoint path must not include scheme, domain, query, or fragment")

    normalized_path = parsed_path.path if parsed_path.path.startswith('/') else '/' + parsed_path.path

    if normalized_path not in allowed_endpoints:
        raise ValueError(f"Unapproved endpoint: {normalized_path}")

    return normalized_path


def parse_and_validate_base_url(base_url, approved_domains=APPROVED_API_DOMAINS):
    """
    Parse and validate the base URL.
    Ensures scheme is http/https and domain is approved.
    Returns the parsed URL.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("Invalid base URL")

    parsed_base = urllib.parse.urlparse(base_url)
    if parsed_base.scheme not in ('http', 'https') or not parsed_base.netloc or parsed_base.hostname is None:
        raise ValueError("Invalid base URL")

    hostname = parsed_base.hostname.lower()
    if hostname not in approved_domains:
        raise ValueError(f"Unapproved domain: {hostname}")

    return parsed_base


def merge_query_params(base_query, params):
    """
    Merge query parameters from the base URL with provided params.
    Provided params take precedence. None values are skipped.
    Returns the encoded query string.
    """
    base_query_map = urllib.parse.parse_qs(base_query, keep_blank_values=True)

    merged_query_map = {}
    for k, v_list in base_query_map.items():
        merged_query_map[k] = [str(v) for v in v_list]

    if params:
        if not isinstance(params, dict):
            raise ValueError("Params must be a dictionary if provided")
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                merged_query_map[k] = [str(item) for item in v]
            else:
                merged_query_map[k] = [str(v)]

    return urllib.parse.urlencode(merged_query_map, doseq=True)


def build_final_url(parsed_base, normalized_path, final_query):
    """
    Construct the final URL from its components.
    """
    return urllib.parse.urlunparse((
        parsed_base.scheme,
        parsed_base.netloc,
        normalized_path,
        '',  # params (deprecated)
        final_query,
        ''   # fragment
    ))


def ensure_final_domain_approved(final_url, approved_domains=APPROVED_API_DOMAINS):
    """
    Verify the constructed URL points to an approved domain.
    """
    parsed_final = urllib.parse.urlparse(final_url)
    hostname = (parsed_final.hostname or '').lower()
    if hostname not in approved_domains:
        raise ValueError("Constructed URL points to an unapproved domain")


def create_api_redirect_link(base_url, path, params=None):
    """
    Create an API redirect URL by merging a base API URL with a defined endpoint and optional query parameters.

    Args:
        base_url (str): The API's base URL (e.g., https://api.myservice.com).
        path (str): The target API endpoint (must be one of ALLOWED_ENDPOINTS).
        params (dict, optional): Optional key-value pairs to include as query parameters.

    Returns:
        str: The assembled API URL.

    Raises:
        ValueError: If the base URL is invalid, or if the domain or endpoint is not approved.
    """
    logger.info("create_api_redirect_link attempt base_url=%r path=%r params=%r", base_url, path, params)
    try:
        normalized_path = validate_and_normalize_path(path, ALLOWED_ENDPOINTS)
        parsed_base = parse_and_validate_base_url(base_url, APPROVED_API_DOMAINS)
        final_query = merge_query_params(parsed_base.query, params)
        final_url = build_final_url(parsed_base, normalized_path, final_query)
        ensure_final_domain_approved(final_url, APPROVED_API_DOMAINS)
        logger.info("create_api_redirect_link success final_url=%s", final_url)
        return final_url
    except Exception:
        logger.exception("create_api_redirect_link failed base_url=%r path=%r params=%r", base_url, path, params)
        raise
