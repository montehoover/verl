from urllib.parse import urlparse

APPROVED_DOMAINS = {'example.com', 'test.com', 'myservice.com'}
ALLOWED_ENDPOINTS = {'/home', '/about', '/contact'}


def validate_url_domain(url: str) -> bool:
    """
    Legacy helper: Returns True if the URL's hostname matches exactly one of the approved domains, else False.
    """
    if not isinstance(url, str) or not url:
        return False

    parsed = urlparse(url)
    if not parsed.netloc:
        parsed = urlparse(f"http://{url}")

    hostname = parsed.hostname
    if not hostname:
        return False

    hostname = hostname.lower().strip(".")
    return hostname in APPROVED_DOMAINS


def validate_url_and_endpoint(url: str, endpoint: str) -> bool:
    """
    Returns True only if:
      - The URL's hostname matches exactly one of the APPROVED_DOMAINS.
      - The provided endpoint is in ALLOWED_ENDPOINTS.
      - The URL's path (ignoring any query parameters) exactly matches the provided endpoint.
    """
    if not isinstance(url, str) or not url:
        return False
    if not isinstance(endpoint, str) or not endpoint:
        return False

    parsed = urlparse(url)
    if not parsed.netloc:
        parsed = urlparse(f"http://{url}")

    hostname = parsed.hostname
    if not hostname:
        return False

    hostname = hostname.lower().strip(".")
    if hostname not in APPROVED_DOMAINS:
        return False

    # Ignore query parameters from URL by using parsed.path only
    url_path = parsed.path or ""

    if endpoint not in ALLOWED_ENDPOINTS:
        return False

    return url_path == endpoint
