import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def construct_url(base_url: str, path: str) -> str:
    """
    Construct a URL by concatenating the base_url with the path.

    Args:
        base_url (str): The base URL.
        path (str): The path to append to the base URL.

    Returns:
        str: The concatenated full URL.
    """
    return f"{base_url}{path}"


def construct_url_with_params(base_url: str, path: str, query_params: dict | None = None) -> str:
    """
    Construct a secure URL by ensuring the base_url uses HTTPS, concatenating with the path,
    and appending URL-encoded query parameters.

    Args:
        base_url (str): The base URL.
        path (str): The path to append to the base URL.
        query_params (dict | None): Dictionary of query parameters to include.

    Returns:
        str: The complete URL with properly encoded query parameters.
    """
    # Ensure base_url starts with https://
    split = urllib.parse.urlsplit(base_url)
    if split.scheme == "https":
        secure_base = base_url
    elif split.scheme == "http":
        secure_base = "https://" + base_url[len("http://"):]
    else:
        # No scheme or a non-HTTP scheme; force HTTPS
        if "://" in base_url:
            secure_base = "https://" + base_url.split("://", 1)[1]
        else:
            secure_base = "https://" + base_url.lstrip("/")

    # Concatenate base and path with exactly one slash between them
    if secure_base.endswith("/") and path.startswith("/"):
        full = secure_base + path.lstrip("/")
    elif not secure_base.endswith("/") and not path.startswith("/"):
        full = secure_base + "/" + path
    else:
        full = secure_base + path

    # Build query string if provided
    if query_params:
        # Filter out None values to avoid "key=None" in the query string
        filtered = {k: v for k, v in query_params.items() if v is not None}
        if filtered:
            query_string = urllib.parse.urlencode(filtered, doseq=True)
            delimiter = "&" if "?" in full else "?"
            full += f"{delimiter}{query_string}"

    return full


def create_api_redirect_link(base_url: str, path: str, params: dict | None = None) -> str:
    """
    Create a redirect URL for approved API domains and endpoints.

    Requirements:
      - Use urllib.parse for URL manipulation.
      - Validate base_url's domain against APPROVED_API_DOMAINS.
      - Validate path against ALLOWED_ENDPOINTS.
      - Append URL-encoded query parameters from params.

    Args:
        base_url (str): The base address (e.g., https://api.myservice.com).
        path (str): The specific API endpoint (must be in ALLOWED_ENDPOINTS).
        params (dict | None): Optional query parameters.

    Returns:
        str: The fully constructed redirect URL.

    Raises:
        ValueError: If the domain or endpoint is not approved/allowed.
    """
    # Validate endpoint
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("Invalid endpoint format; path must be a string starting with '/'.")
    if path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint not allowed: {path}")

    # Parse base URL and extract/normalize hostname
    parsed = urllib.parse.urlparse(base_url)
    hostname = parsed.hostname

    # Handle bare domains without scheme (e.g., "api.myservice.com")
    if not hostname:
        candidate = base_url.strip()
        candidate = candidate.split("://", 1)[-1] if "://" in candidate else candidate
        candidate = candidate.lstrip("/").split("/", 1)[0]
        hostname = candidate or None

    if not hostname or hostname.lower() not in APPROVED_API_DOMAINS:
        raise ValueError("Base URL domain is not approved.")

    domain = hostname.lower()

    # Default to https if scheme is missing or unrecognized; otherwise preserve http/https
    scheme = parsed.scheme if parsed.scheme in ("http", "https") else "https"

    # For strictness, drop any userinfo/port/path/query from base_url; only approved domain is used
    netloc = domain

    # Build query string
    query = ""
    if params:
        filtered = {str(k): v for k, v in params.items() if v is not None}
        if filtered:
            query = urllib.parse.urlencode(filtered, doseq=True)

    # Assemble final URL using urllib.parse
    return urllib.parse.urlunparse((scheme, netloc, path, "", query, ""))
