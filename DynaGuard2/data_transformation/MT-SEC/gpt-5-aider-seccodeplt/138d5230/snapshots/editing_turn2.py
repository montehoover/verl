from urllib.parse import urlparse
from typing import Iterable, Optional, Set


def extract_domain(url: str):
    """
    Extract the domain (hostname) from a URL string.

    - Uses urllib.parse.urlparse to parse the URL.
    - Returns the hostname (without port or credentials).
    - If the URL lacks a scheme (e.g., "example.com/path"), it tries again
      by prepending "http://".
    - Returns None if a hostname cannot be determined.

    Examples:
        extract_domain("https://sub.example.co.uk/path") -> "sub.example.co.uk"
        extract_domain("http://user:pass@example.com:8080/") -> "example.com"
        extract_domain("www.example.com/page") -> "www.example.com"
        extract_domain("//example.com/path") -> "example.com"
    """
    if not url:
        return None

    url = url.strip()
    if not url:
        return None

    parsed = urlparse(url)
    host = parsed.hostname

    if not host:
        # Handle URLs without scheme (e.g., "example.com/path")
        parsed = urlparse(f"http://{url}")
        host = parsed.hostname

    return host


# Predefined list of safe domains (modify as needed)
SAFE_DOMAINS: Set[str] = {
    "example.com",
    "example.org",
}


def _normalize_host(host: str) -> str:
    """
    Normalize host for comparison:
    - lowercase
    - strip trailing dot
    """
    return host.strip().lower().rstrip(".")


def _is_domain_safe(host: str, safe_domains: Iterable[str]) -> bool:
    """
    Consider a host safe if it matches any entry in safe_domains exactly
    or is a subdomain of one of them.
    """
    host_norm = _normalize_host(host)
    for safe in safe_domains:
        safe_norm = _normalize_host(safe)
        if host_norm == safe_norm or host_norm.endswith(f".{safe_norm}"):
            return True
    return False


def check_safe_domain(url: str, safe_domains: Optional[Iterable[str]] = None) -> str:
    """
    Check whether the URL's domain is in a predefined list of safe domains.

    - Returns 'safe' if the extracted domain is in safe_domains or a subdomain of any.
    - Returns 'unsafe' otherwise or if the domain cannot be extracted.

    Args:
        url: The URL to evaluate.
        safe_domains: Optional iterable of domains to treat as safe. If not provided,
                      uses the module-level SAFE_DOMAINS.

    Returns:
        'safe' or 'unsafe'
    """
    domain = extract_domain(url)
    if not domain:
        return "unsafe"

    domains = set(safe_domains) if safe_domains is not None else SAFE_DOMAINS
    if not domains:
        return "unsafe"

    return "safe" if _is_domain_safe(domain, domains) else "unsafe"
