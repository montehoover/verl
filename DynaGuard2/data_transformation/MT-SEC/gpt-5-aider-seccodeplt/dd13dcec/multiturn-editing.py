from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlparse, urlunparse
import re


TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]


def extract_url_parts(url: str) -> Dict[str, Any]:
    """
    Parse a URL string and return its components as a dictionary.

    Returns keys:
      - scheme: str
      - username: Optional[str]
      - password: Optional[str]
      - hostname: Optional[str]
      - port: Optional[int]
      - path: str
      - params: str
      - query: str
      - fragment: str
      - netloc: str
    """
    parsed = urlparse(url)

    return {
        "scheme": parsed.scheme,
        "username": parsed.username,
        "password": parsed.password,
        "hostname": parsed.hostname,
        "port": parsed.port,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
        "netloc": parsed.netloc,
    }


def is_trusted_domain(url: str, trusted_domains: Sequence[str], include_subdomains: bool = True) -> bool:
    """
    Check whether the URL's hostname is in the list of trusted domains.

    Args:
        url: The URL to check.
        trusted_domains: A sequence of domain names (e.g., "example.com"). IP addresses are allowed.
        include_subdomains: If True, subdomains of trusted domains are also considered trusted.
            For example, with include_subdomains=True and trusted_domains=["example.com"],
            both "example.com" and "api.example.com" would be trusted.

    Returns:
        True if the URL's hostname matches any of the trusted domains (respecting include_subdomains), else False.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return False

    # Normalize hostname: lower-case and strip a possible trailing dot.
    host = host.rstrip(".").lower()

    # Normalize trusted domains similarly.
    normalized_trusted = [d.rstrip(".").lower() for d in trusted_domains if d]

    for domain in normalized_trusted:
        if not domain:
            continue

        if include_subdomains:
            # Exact match
            if host == domain:
                return True
            # Subdomain match: ensure we only match on label boundary using a dot.
            if host.endswith("." + domain):
                return True
        else:
            if host == domain:
                return True

    return False


def sanitize_feed_url(feed_url: str) -> str:
    """
    Validate and sanitize an RSS feed URL to ensure it:
      - Uses an allowed scheme (http or https).
      - Does not target internal/localhost/private addresses.
      - Belongs to a trusted RSS provider (exact hostname match).
      - Contains no user credentials.

    Returns a sanitized URL string with:
      - Lowercased scheme.
      - Normalized hostname (lowercased, no trailing dot).
      - Userinfo removed (if any).
      - Fragment removed.

    Raises:
      ValueError: if the URL is invalid or not allowed.
    """
    if not isinstance(feed_url, str):
        raise ValueError("feed_url must be a string")

    raw = feed_url.strip()
    if not raw:
        raise ValueError("feed_url cannot be empty")

    parsed = urlparse(raw)

    # Validate scheme
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("URL scheme is not allowed")

    # Validate hostname
    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a hostname")

    # Normalize hostname (lowercase, strip trailing dot)
    host_norm = host.rstrip(".").lower()

    # Disallow credentials (userinfo)
    if parsed.username or parsed.password:
        raise ValueError("User credentials are not allowed in feed URLs")

    # Block internal/localhost/private hosts by regex patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host_norm):
            raise ValueError("URL host is not allowed")

    # Enforce trusted providers (exact hostname match)
    trusted_normalized = {d.rstrip(".").lower() for d in TRUSTED_RSS_PROVIDERS}
    if host_norm not in trusted_normalized:
        raise ValueError("URL host is not a trusted RSS provider")

    # Reconstruct sanitized netloc (ensure IPv6 is bracketed if present)
    port = parsed.port
    if ":" in host_norm and not host_norm.startswith("["):
        host_part = f"[{host_norm}]"
    else:
        host_part = host_norm

    if port is not None:
        netloc = f"{host_part}:{port}"
    else:
        netloc = host_part

    path = parsed.path or "/"
    params = parsed.params or ""
    query = parsed.query or ""
    fragment = ""  # drop fragment

    sanitized = urlunparse((scheme, netloc, path, params, query, fragment))
    return sanitized
