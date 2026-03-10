import re
from urllib.parse import urlparse


# Hardcoded list of trusted domains (normalize to lowercase, no trailing dots)
TRUSTED_DOMAINS = {
    "example.com",
    "mycompany.com",
    "trusted.org",
}


def _normalize_hostname(host: str) -> str:
    # Lowercase and remove any trailing dot for consistent comparisons
    return host.strip().lower().rstrip(".")


def extract_domain(url: str) -> str:
    """
    Extract the domain (host) from a URL string.

    Examples:
      - "https://www.example.com/path" -> "www.example.com"
      - "example.com/path" -> "example.com"
      - "http://user:pass@example.com:8080" -> "example.com"
      - "http://[2001:db8::1]:8080/" -> "2001:db8::1"

    Returns an empty string if no domain can be determined.
    """
    if not isinstance(url, str) or not url:
        return ""

    parsed = urlparse(url)

    # Handle URLs without a scheme like "example.com/path"
    if not parsed.netloc and parsed.path:
        parsed = urlparse("//" + url)

    host = parsed.hostname  # Strips credentials and port; lowercases host
    if not host:
        return ""

    # Remove any trailing dot from the hostname
    if host.endswith("."):
        host = host[:-1]

    return host


# Pre-normalize trusted domains for faster comparisons
_NORMALIZED_TRUSTED = {_normalize_hostname(d) for d in TRUSTED_DOMAINS if d}


def is_trusted_domain(url: str) -> bool:
    """
    Determine whether the URL's domain is trusted.

    A domain is considered trusted if it exactly matches a trusted domain
    or is a subdomain of a trusted domain.

    Examples:
      - "https://sub.example.com/page" with "example.com" trusted -> True
      - "https://example.com" with "example.com" trusted -> True
      - "https://nottrusted.io" -> False
    """
    domain = extract_domain(url)
    if not domain:
        return False

    domain = _normalize_hostname(domain)
    for trusted in _NORMALIZED_TRUSTED:
        if domain == trusted or domain.endswith("." + trusted):
            return True
    return False


# RSS validation configuration
TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]

_NORMALIZED_TRUSTED_RSS = {_normalize_hostname(h) for h in TRUSTED_RSS_PROVIDERS}
_INTERNAL_IP_REGEXES = [re.compile(p, re.IGNORECASE) for p in INTERNAL_IP_PATTERNS]
_ALLOWED_SCHEMES_SET = {s.lower() for s in ALLOWED_SCHEMES}


def _is_internal_host(host: str) -> bool:
    hn = _normalize_hostname(host)
    return any(r.fullmatch(hn) for r in _INTERNAL_IP_REGEXES)


def _is_trusted_provider(host: str) -> bool:
    hn = _normalize_hostname(host)
    for trusted in _NORMALIZED_TRUSTED_RSS:
        if hn == trusted or hn.endswith("." + trusted):
            return True
    return False


def validate_rss_link(rss_link: str) -> str:
    """
    Validate and sanitize an RSS feed URL.

    - Must use an allowed scheme (http or https). If missing, defaults to https.
    - Host must be in the list of trusted providers (or their subdomains).
    - Must not target internal networks (e.g., localhost, 127.0.0.0/8, 10.0.0.0/8, 192.168.0.0/16).
    - Returns a sanitized URL with:
        * lowercase scheme and host
        * no credentials
        * preserved path and query
        * no fragment or params

    Raises ValueError on any validation failure.
    """
    if not isinstance(rss_link, str) or not rss_link.strip():
        raise ValueError("RSS link must be a non-empty string")

    candidate = rss_link.strip()

    parsed = urlparse(candidate)
    # Handle URLs without a scheme like "rss.trustedsource.com/feed"
    if not parsed.netloc and parsed.path:
        parsed = urlparse("//" + candidate)

    host = parsed.hostname  # already lowercased
    if not host:
        raise ValueError("RSS link must include a hostname")

    # Disallow internal hosts
    if _is_internal_host(host):
        raise ValueError("RSS link points to an internal host")

    # Ensure host is a trusted provider (or subdomain)
    if not _is_trusted_provider(host):
        raise ValueError("RSS link host is not an approved provider")

    # Validate or default scheme
    scheme = (parsed.scheme or "https").lower()
    if scheme not in _ALLOWED_SCHEMES_SET:
        raise ValueError("RSS link must use http or https")

    # Sanitize netloc: exclude credentials, normalize host casing, keep port if provided
    hostname = _normalize_hostname(host)
    port = parsed.port
    if ":" in hostname and not hostname.startswith("["):
        host_for_netloc = f"[{hostname}]"
    else:
        host_for_netloc = hostname
    netloc = f"{host_for_netloc}:{port}" if port else host_for_netloc

    # Build sanitized URL without params or fragment
    path = parsed.path or ""
    query = parsed.query or ""
    sanitized = f"{scheme}://{netloc}{path}"
    if query:
        sanitized += f"?{query}"

    return sanitized
