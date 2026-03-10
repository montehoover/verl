import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]


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

    # Block internal network hosts by regex patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            raise ValueError("Internal network addresses are not allowed.")

    # Allow only approved domains (exact match or subdomain)
    allowed = False
    for domain in ALLOWED_API_DOMAINS:
        d = domain.lower().rstrip(".")
        if host == d or host.endswith("." + d):
            allowed = True
            break

    if not allowed:
        raise ValueError("Unapproved domain.")

    return url
