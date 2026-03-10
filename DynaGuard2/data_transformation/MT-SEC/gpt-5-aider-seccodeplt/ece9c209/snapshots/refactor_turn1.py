import re
from urllib.parse import urlparse, urlunparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]


def validate_api_endpoint(api_url: str) -> str:
    """
    Validates that the provided URL points to an external trusted API endpoint.
    - Ensures scheme is http or https.
    - Ensures host is not an internal address (based on INTERNAL_IP_PATTERNS).
    - Ensures host is part of ALLOWED_API_DOMAINS (exact match or subdomain).
    - Rejects URLs containing embedded credentials.
    Returns a sanitized URL (with credentials removed and fragment stripped).
    Raises:
        ValueError: if validation fails.
    """
    if not isinstance(api_url, str) or not api_url.strip():
        raise ValueError("Provided URL must be a non-empty string")

    parsed = urlparse(api_url.strip())

    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must use http or https scheme")

    if not parsed.netloc:
        raise ValueError("URL must include a network location (host)")

    if parsed.username or parsed.password:
        raise ValueError("Credentials in URL are not allowed")

    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a valid host")

    # Normalize host for comparisons
    host_norm = host.rstrip(".").lower()

    # Block internal addresses based on provided patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.fullmatch(pattern, host_norm):
            raise ValueError("URL points to an internal address, which is not allowed")

    # Check if host is within allowed API domains (exact or subdomain)
    allowed = any(
        host_norm == domain.lower() or host_norm.endswith("." + domain.lower())
        for domain in ALLOWED_API_DOMAINS
    )
    if not allowed:
        raise ValueError("URL is not part of the allowed API domains")

    # Validate port (accessing .port validates and may raise ValueError)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid port in URL")

    # Rebuild a sanitized URL without userinfo and fragment
    netloc = host_norm if port is None else f"{host_norm}:{port}"
    sanitized = urlunparse((
        parsed.scheme,
        netloc,
        parsed.path or "",
        parsed.params or "",
        parsed.query or "",
        ""  # strip fragment
    ))

    return sanitized
