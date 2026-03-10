from urllib.parse import urlparse, urlunparse
import posixpath
import re

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

# Pre-normalize allowed domains to ASCII (punycode) lowercase for consistent comparison
_ALLOWED_DOMAINS_ASCII = []
for _d in ALLOWED_REDIRECT_DOMAINS:
    try:
        _ALLOWED_DOMAINS_ASCII.append(_d.encode("idna").decode("ascii").lower())
    except Exception:
        # Skip domains that cannot be encoded (should not happen for valid domains)
        pass

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F\x7F]")


def _sanitize_component(value: str) -> str:
    # Remove control characters to prevent header injection or other abuses
    return _CONTROL_CHARS_RE.sub("", value)


def _domain_allowed(host_ascii: str) -> bool:
    host_ascii = (host_ascii or "").lower()
    for d in _ALLOWED_DOMAINS_ASCII:
        if host_ascii == d or host_ascii.endswith("." + d):
            return True
    return False


def user_defined_redirect(redirect_url: str) -> str:
    """
    Accept a user-provided URL and return a sanitized, safe-to-redirect absolute URL.

    Rules:
    - Only http/https schemes are allowed (default to https if missing and host present).
    - The URL must be absolute and contain a host; relative URLs are rejected.
    - Username/password (userinfo) are not allowed.
    - Host must be in ALLOWED_REDIRECT_DOMAINS (exact match or subdomain).
    - Control characters are stripped from path/query/params.
    - Fragment is removed.
    """
    if not isinstance(redirect_url, str):
        raise ValueError("URL must be a string")

    s = redirect_url.strip()
    if not s:
        raise ValueError("URL is empty")

    parsed = urlparse(s)

    # If both scheme and netloc are missing, try to interpret as scheme-less absolute URL
    if not parsed.scheme and not parsed.netloc:
        if s.startswith("/"):
            # Relative URLs are not allowed to avoid open redirects within current site
            raise ValueError("Relative URLs are not allowed")
        # Try to treat as bare domain/path by prefixing https://
        attempt = urlparse("https://" + s)
        if not attempt.netloc:
            raise ValueError("Invalid URL")
        parsed = attempt

    # Block userinfo (username/password) to prevent deceptive URLs like user@host@evil.com
    if parsed.username or parsed.password:
        raise ValueError("User info in URL is not allowed")

    # Enforce scheme
    scheme = (parsed.scheme or "https").lower()
    if scheme not in ("http", "https"):
        raise ValueError("Unsupported URL scheme")

    host = parsed.hostname  # Excludes userinfo and port
    if not host:
        raise ValueError("Invalid URL: missing host")

    # Normalize host to ASCII punycode for consistent comparison and output
    is_ipv6 = ":" in host and not host.startswith("xn--")
    try:
        host_ascii = host if is_ipv6 else host.encode("idna").decode("ascii")
    except Exception:
        raise ValueError("Invalid host")

    # Check against allowed domains (allow exact match or subdomain)
    if not _domain_allowed(host_ascii):
        raise ValueError("Domain not allowed")

    # Rebuild netloc without userinfo
    port = parsed.port
    if is_ipv6:
        netloc_host = f"[{host_ascii}]"
    else:
        netloc_host = host_ascii.lower()
    netloc = f"{netloc_host}:{port}" if port else netloc_host

    # Sanitize components
    path = _sanitize_component(parsed.path or "")
    params = _sanitize_component(parsed.params or "")
    query = _sanitize_component(parsed.query or "")

    # Normalize path to remove dot-segments while preserving leading slash if present
    if path:
        leading_slash = path.startswith("/")
        norm_path = posixpath.normpath(path)
        if leading_slash and not norm_path.startswith("/"):
            norm_path = "/" + norm_path
        if norm_path == ".":
            norm_path = "/"
        path = norm_path

    # Drop fragment for safety
    fragment = ""

    sanitized_url = urlunparse((scheme, netloc, path, params, query, fragment))
    return sanitized_url
