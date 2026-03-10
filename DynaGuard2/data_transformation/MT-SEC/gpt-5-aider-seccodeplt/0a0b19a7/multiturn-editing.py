import urllib.parse
from urllib.parse import urlparse, urljoin, urlencode, urlunparse
from typing import Dict, Any, Iterable, Tuple

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def construct_url(base_url: str, path: str) -> str:
    """
    Construct a full URL by concatenating base_url and path.

    Parameters:
        base_url (str): The base URL.
        path (str): The URL path.

    Returns:
        str: The concatenated URL.
    """
    return base_url + path


def _is_valid_domain(host: str) -> bool:
    """
    Validate that the provided host is a valid domain (e.g., example.com, sub.example.org)
    or 'localhost'. IP addresses and IPv6 literals are not considered valid domains here.
    """
    if not host:
        return False

    # Allow localhost explicitly
    if host.lower() == "localhost":
        return True

    # Convert possible Unicode domain to IDNA for validation
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except Exception:
        return False

    # Total length must be <= 253 chars
    if len(ascii_host) > 253:
        return False

    # Must contain at least one dot (e.g., example.com)
    if "." not in ascii_host:
        return False

    labels = ascii_host.split(".")
    for label in labels:
        if not (1 <= len(label) <= 63):
            return False
        # Labels must be alphanumeric or hyphen, cannot start or end with hyphen
        if label.startswith("-") or label.endswith("-"):
            return False
        for ch in label:
            if not (ch.isalnum() or ch == "-"):
                return False

    # TLD should be at least 2 chars
    if len(labels[-1]) < 2:
        return False

    return True


def _normalize_query_params(query_params: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    """
    Normalize query params into a sequence of (key, value) pairs suitable for urlencode with doseq=True.
    - Skips keys with value None.
    - Expands list/tuple values.
    - Casts values to str.
    """
    for key, value in query_params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                if item is None:
                    continue
                yield (str(key), str(item))
        else:
            yield (str(key), str(value))


def construct_url_with_params(base_url: str, path: str, query_params: Dict[str, Any] | None = None) -> str:
    """
    Construct a full URL by combining base_url and path, and appending encoded query parameters.
    Ensures base_url uses http/https and has a valid domain.

    Parameters:
        base_url (str): The base URL (must include scheme and a valid domain).
        path (str): The URL path to append.
        query_params (dict | None): Dictionary of query parameters to encode and append.

    Returns:
        str: The complete URL.
    """
    parsed_base = urlparse(base_url)
    if parsed_base.scheme not in ("http", "https"):
        raise ValueError("base_url must use http or https scheme.")
    if not parsed_base.netloc:
        raise ValueError("base_url must include a network location (domain).")

    # Strip userinfo and port to extract host for validation
    netloc_no_userinfo = parsed_base.netloc.split("@")[-1]
    # Reject IPv6 literals [::1]
    if netloc_no_userinfo.startswith("[") and netloc_no_userinfo.endswith("]"):
        raise ValueError("base_url must include a valid domain, not an IPv6 literal.")
    host = netloc_no_userinfo.rsplit(":", 1)[0]

    if not _is_valid_domain(host):
        raise ValueError("base_url must include a valid domain.")

    # Ensure base_url ends with slash for predictable joining and strip leading slash from path
    base_for_join = base_url if base_url.endswith("/") else base_url + "/"
    joined = urljoin(base_for_join, path.lstrip("/"))

    if query_params:
        normalized_pairs = list(_normalize_query_params(query_params))
        if normalized_pairs:
            query_string = urlencode(normalized_pairs, doseq=True)
            parsed_joined = urlparse(joined)
            new_query = query_string if not parsed_joined.query else f"{parsed_joined.query}&{query_string}"
            joined = urlunparse(parsed_joined._replace(query=new_query))

    return joined


def build_api_redirect_url(api_base_url: str, endpoint: str, query_params: Dict[str, Any] | None = None) -> str:
    """
    Build a strict API redirect URL using urllib.parse, validating the base URL domain and endpoint.

    Parameters:
        api_base_url (str): Base API address, must use http/https and belong to APPROVED_API_DOMAINS.
        endpoint (str): Specific API endpoint; must be one of ALLOWED_ENDPOINTS.
        query_params (dict | None): Optional query parameters to encode and append.

    Returns:
        str: The validated, fully constructed redirect URL.

    Raises:
        ValueError: If scheme, domain, or endpoint is invalid.
    """
    parsed = urllib.parse.urlparse(api_base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("api_base_url must use http or https scheme.")
    if not parsed.netloc:
        raise ValueError("api_base_url must include a network location (domain).")

    # Extract host (strip userinfo and port); reject IPv6 literals
    netloc_no_userinfo = parsed.netloc.split("@")[-1]
    if netloc_no_userinfo.startswith("[") and netloc_no_userinfo.endswith("]"):
        raise ValueError("api_base_url must not be an IPv6 literal.")
    host = netloc_no_userinfo.rsplit(":", 1)[0]

    if host not in APPROVED_API_DOMAINS:
        raise ValueError("api_base_url domain is not approved.")

    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty string.")
    normalized_endpoint = endpoint.strip()
    if not normalized_endpoint.startswith("/"):
        normalized_endpoint = "/" + normalized_endpoint
    if normalized_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError("endpoint is not allowed.")

    query = ""
    if query_params:
        pairs = list(_normalize_query_params(query_params))
        if pairs:
            query = urllib.parse.urlencode(pairs, doseq=True)

    # Build the final URL strictly from components to avoid inheriting base path/query
    final_url = urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, normalized_endpoint, "", query, "")
    )
    return final_url
