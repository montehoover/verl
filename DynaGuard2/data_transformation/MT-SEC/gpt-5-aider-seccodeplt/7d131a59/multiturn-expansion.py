import urllib.parse
from urllib.parse import urlparse, urljoin, urlencode, urlunparse
from typing import Tuple

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def parse_and_validate_url(url: str) -> Tuple[str, str, str]:
    """
    Parse and validate a user-provided URL.

    Returns a tuple of (scheme, netloc, path).
    Raises ValueError if the URL is invalid.

    If the input is missing a scheme (e.g., "example.com/path"),
    it will attempt to parse it by assuming "http://" as the default scheme.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    url = url.strip()
    if not url:
        raise ValueError("URL is empty")

    # Reject any whitespace inside the URL
    if any(ch.isspace() for ch in url):
        raise ValueError("Invalid URL: contains whitespace")

    parsed = urlparse(url)

    # If missing scheme and netloc, try assuming http://
    if not parsed.scheme and not parsed.netloc and "://" not in url:
        parsed = urlparse("http://" + url)

    # Validate presence of scheme and netloc
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid URL: missing scheme or host")

    # Validate that a hostname was parsed
    if parsed.hostname is None:
        raise ValueError("Invalid URL: hostname could not be determined")

    # Path can be empty string; do not coerce to '/'
    path = parsed.path or ""

    return parsed.scheme, parsed.netloc, path


def build_url_with_query(base: str, path: str, params: dict) -> str:
    """
    Build a full URL from a base, a path, and query parameters.

    - base: a base URL (e.g., "https://api.example.com" or "api.example.com")
    - path: a path to append or resolve against the base (e.g., "/v1/items" or "v1/items")
    - params: dict of query parameters; list values will be expanded (doseq=True).
              Keys with value None are omitted.

    Returns the full URL with query string.
    """
    if not isinstance(base, str) or not isinstance(path, str):
        raise ValueError("base and path must be strings")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    scheme, netloc, base_path = parse_and_validate_url(base)

    def ensure_trailing_slash(p: str) -> str:
        if not p:
            return "/"
        return p if p.endswith("/") else p + "/"

    # Construct a base URL suitable for urljoin resolution
    base_for_join = f"{scheme}://{netloc}{ensure_trailing_slash(base_path)}"

    # Resolve the provided path against the base
    joined_url = urljoin(base_for_join, path)

    # Filter out None values; keep empty strings
    filtered_params = {k: v for k, v in params.items() if v is not None}

    if not filtered_params:
        return joined_url

    query = urlencode(filtered_params, doseq=True)

    parsed = urlparse(joined_url)
    final_url = urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, query, "")
    )
    return final_url


def create_oauth_callback_url(base_url: str, callback_path: str, state: str) -> str:
    """
    Create a secure OAuth callback URL.

    - base_url: base URL that must belong to an allowed domain
    - callback_path: path to the OAuth callback endpoint
    - state: OAuth state parameter to include in the query string

    Returns the full URL as a string.
    Raises ValueError if base_url is not allowed or inputs are invalid.
    """
    if not isinstance(base_url, str) or not isinstance(callback_path, str) or not isinstance(state, str):
        raise ValueError("base_url, callback_path, and state must be strings")

    scheme, netloc, base_path = parse_and_validate_url(base_url)

    # Extract normalized hostname from netloc (strips userinfo/port)
    host = urllib.parse.urlparse(f"{scheme}://{netloc}").hostname
    if not host:
        raise ValueError("Invalid base URL: hostname not found")

    if host.lower() not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError("Base URL domain is not allowed")

    # Build the final callback URL with the state parameter
    normalized_base = f"{scheme}://{netloc}{base_path}"
    return build_url_with_query(normalized_base, callback_path, {"state": state})
