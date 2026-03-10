import re
from urllib.parse import urlparse, urlunparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]


def _normalize_host(host: str) -> str:
    """
    Normalize a hostname for consistent comparison:
    - Remove any trailing dot.
    - Convert to lowercase.
    Pure function: output depends only on input, no side effects.
    """
    return (host or "").rstrip(".").lower()


def is_internal_host(host: str, internal_ip_patterns: list[str]) -> bool:
    """
    Check whether the given host matches any internal network patterns.
    Pure function: relies solely on inputs and produces no side effects.
    """
    host_norm = _normalize_host(host)
    for pattern in internal_ip_patterns:
        if re.fullmatch(pattern, host_norm):
            return True
    return False


def is_allowed_api_domain(host: str, allowed_domains: list[str]) -> bool:
    """
    Check whether the given host is an exact match of, or a subdomain of,
    any domain in allowed_domains.
    Pure function: relies solely on inputs and produces no side effects.
    """
    host_norm = _normalize_host(host)
    for domain in allowed_domains:
        domain_norm = _normalize_host(domain)
        if host_norm == domain_norm or host_norm.endswith("." + domain_norm):
            return True
    return False


def validate_api_endpoint(api_url: str) -> str:
    """
    Validates that the provided URL points to an external trusted API endpoint.

    Validation steps:
    1) Ensure api_url is a non-empty string.
    2) Parse the URL and verify the scheme is http or https.
    3) Ensure a network location (host) is present.
    4) Forbid embedded credentials (userinfo).
    5) Extract and normalize the host.
    6) Block internal/localhost/private network hosts.
    7) Ensure host belongs to allowed API domains (exact or subdomain).
    8) Validate the port (if present).
    9) Return a sanitized URL with:
       - normalized host,
       - preserved scheme/path/query/params,
       - explicit port if present,
       - no userinfo and no fragment.

    Raises:
        ValueError: if any validation step fails.
    """
    # 1) Basic input validation
    if not isinstance(api_url, str) or not api_url.strip():
        raise ValueError("Provided URL must be a non-empty string")

    # 2) Parse URL and enforce allowed schemes
    parsed = urlparse(api_url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must use http or https scheme")

    # 3) Ensure host is present
    if not parsed.netloc:
        raise ValueError("URL must include a network location (host)")

    # 4) Forbid userinfo (credentials) in the URL
    if parsed.username or parsed.password:
        raise ValueError("Credentials in URL are not allowed")

    # 5) Extract and normalize the host
    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a valid host")
    host_norm = _normalize_host(host)

    # 6) Block internal hosts (localhost, loopback, private ranges)
    if is_internal_host(host_norm, INTERNAL_IP_PATTERNS):
        raise ValueError("URL points to an internal address, which is not allowed")

    # 7) Ensure host is within allowed API domains (exact or subdomain)
    if not is_allowed_api_domain(host_norm, ALLOWED_API_DOMAINS):
        raise ValueError("URL is not part of the allowed API domains")

    # 8) Validate port; accessing .port will validate and may raise ValueError
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid port in URL")

    # 9) Rebuild a sanitized URL without userinfo and fragment,
    #    using the normalized host and preserving other parts.
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
