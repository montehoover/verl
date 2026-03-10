import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

COMPILED_INTERNAL_IP_PATTERNS = [re.compile(p) for p in INTERNAL_IP_PATTERNS]


def is_internal_host(host: str) -> bool:
    """
    Determine whether the given host matches internal network patterns.

    Args:
        host: A hostname or IP address (string).

    Returns:
        True if the host is considered internal, False otherwise.
    """
    normalized_host = (host or "").lower().rstrip(".")
    for pattern in COMPILED_INTERNAL_IP_PATTERNS:
        if pattern.match(normalized_host):
            return True
    return False


def is_allowed_domain(host: str, allowed_domains=None) -> bool:
    """
    Check whether the given host is within the allowed API domains
    (exact match or a subdomain).

    Args:
        host: A hostname to check.
        allowed_domains: Optional list of allowed base domains. Defaults to ALLOWED_API_DOMAINS.

    Returns:
        True if the host is allowed, False otherwise.
    """
    if allowed_domains is None:
        allowed_domains = ALLOWED_API_DOMAINS

    normalized_host = (host or "").lower().rstrip(".")
    if not normalized_host:
        return False

    for domain in allowed_domains:
        d = (domain or "").lower().rstrip(".")
        if not d:
            continue
        if normalized_host == d or normalized_host.endswith("." + d):
            return True
    return False


def enforce_api_security(api_endpoint: str) -> str:
    """
    Ensure the given URL points to a secure, external API endpoint.
    - Requires HTTPS scheme.
    - Disallows credentials in the URL.
    - Disallows internal network hosts (localhost, loopback, private ranges).
    - Allows only hosts within ALLOWED_API_DOMAINS (exact or subdomain).

    Returns:
        The original URL if allowed.

    Raises:
        ValueError: if the URL is invalid or not permitted.
    """
    if not isinstance(api_endpoint, str):
        raise ValueError("URL must be a string.")

    url = api_endpoint.strip()
    if not url:
        raise ValueError("URL cannot be empty.")

    parsed = urlparse(url)

    # Require HTTPS and a valid netloc/host
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError("Only HTTPS endpoints with a valid host are permitted.")

    # Disallow URLs containing credentials
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Credentials in URL are not allowed.")

    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError("URL must contain a valid hostname.")

    # Block internal network hosts
    if is_internal_host(host):
        raise ValueError("Internal network addresses are not allowed.")

    # Allow only approved domains (exact match or subdomain)
    if not is_allowed_domain(host):
        raise ValueError("Unapproved domain.")

    return url
