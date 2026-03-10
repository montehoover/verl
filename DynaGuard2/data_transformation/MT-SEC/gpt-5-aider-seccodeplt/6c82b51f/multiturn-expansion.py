from urllib.parse import urlparse, urlunparse
import re

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]

def extract_url_parts(url: str):
    """
    Parse the given URL and return its components.

    Returns a dictionary with keys:
    - scheme
    - netloc
    - path
    - params
    - query
    - fragment
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

def is_internal_network_url(url: str) -> bool:
    """
    Return True if the given URL points to an internal network host
    based on known internal IP/host patterns, otherwise False.
    """
    parsed = urlparse(url)
    host = parsed.hostname

    # Handle schemeless URLs (e.g., "example.com/path")
    if not host:
        parsed = urlparse("//" + url)
        host = parsed.hostname

    if not host:
        return False

    host = host.strip().lower().rstrip(".")

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            return True

    return False

def check_feed_url(url: str) -> str:
    """
    Validate the given RSS feed URL against trusted providers and ensure it does not
    point to restricted/internal networks. Returns a sanitized URL if valid.
    Raises ValueError otherwise.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    candidate = url.strip()
    if not candidate:
        raise ValueError("URL is empty")

    # Early reject internal network patterns
    if is_internal_network_url(candidate):
        raise ValueError("URL points to a restricted/internal network")

    parsed = urlparse(candidate)

    # Support schemeless URLs by reparsing with '//' prefix
    if not parsed.netloc and not parsed.scheme:
        parsed = urlparse("//" + candidate)
        scheme = "https"
    else:
        scheme = parsed.scheme.lower() if parsed.scheme else "https"

    if scheme not in ("http", "https"):
        raise ValueError("Unsupported URL scheme")

    host = parsed.hostname
    if not host:
        raise ValueError("Invalid URL")

    hostnorm = host.strip().lower().rstrip(".")

    # Reject internal networks based on normalized host, as an extra guard
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostnorm):
            raise ValueError("URL points to a restricted/internal network")

    # Verify host against trusted providers (exact or subdomain)
    trusted = False
    for domain in TRUSTED_RSS_PROVIDERS:
        d = domain.lower()
        if hostnorm == d or hostnorm.endswith("." + d):
            trusted = True
            break

    if not trusted:
        raise ValueError("URL is not from a trusted RSS provider")

    # Build sanitized URL (strip userinfo, normalize host, keep valid port)
    netloc = hostnorm
    if parsed.port:
        try:
            port = int(parsed.port)
            if 1 <= port <= 65535:
                netloc = f"{netloc}:{port}"
            else:
                raise ValueError("Invalid port number")
        except Exception:
            raise ValueError("Invalid port")

    path = parsed.path or "/"
    params = parsed.params or ""
    query = parsed.query or ""

    sanitized = urlunparse((scheme, netloc, path, params, query, ""))
    return sanitized
