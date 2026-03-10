import re
from urllib.parse import urlparse

def extract_url_parts(url: str):
    """
    Parse a URL into its components.

    Returns a dict with keys: scheme, netloc, path, params, query, fragment.
    """
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
    }

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
_COMPILED_INTERNAL_IP_PATTERNS = [re.compile(pat, re.IGNORECASE) for pat in INTERNAL_IP_PATTERNS]

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]

def _extract_hostname(url: str) -> str:
    """
    Extract hostname from a URL, even if the scheme is missing.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        # Try parsing as scheme-relative URL to capture netloc as hostname
        parsed_alt = urlparse(f"//{url}")
        host = parsed_alt.hostname
    return host or ""

def is_unsafe_url(url: str) -> bool:
    """
    Return True if the URL host matches internal IP/hostname patterns.
    """
    host = _extract_hostname(url).lower()
    if not host:
        return False
    for regex in _COMPILED_INTERNAL_IP_PATTERNS:
        if regex.match(host):
            return True
    return False

def check_rss_source(url: str) -> str:
    """
    Validate that the URL is from a trusted RSS provider and not an internal address.
    Returns the URL if valid; raises ValueError otherwise.
    """
    if is_unsafe_url(url):
        raise ValueError("Unsafe URL: internal network addresses are not allowed")

    host = _extract_hostname(url).lower()
    if not host:
        raise ValueError("Invalid URL: could not determine hostname")

    if host not in {h.lower() for h in TRUSTED_RSS_PROVIDERS}:
        raise ValueError("Untrusted RSS provider")

    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("Invalid URL scheme: only http and https are allowed")

    return url
