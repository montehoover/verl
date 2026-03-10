from urllib.parse import urlparse
from typing import Iterable

def extract_domain(url: str) -> str:
    """
    Extract the domain (hostname) from a URL string.

    Uses urllib.parse.urlparse to handle parsing. Returns an empty string if a domain
    cannot be determined.

    Examples:
      - "https://user:pass@example.com:8080/path" -> "example.com"
      - "http://127.0.0.1:5000" -> "127.0.0.1"
      - "example.com/path" -> "example.com"
      - "[2001:db8::1]:8080" -> "2001:db8::1"
    """
    if url is None:
        return ""

    u = url.strip()
    if not u:
        return ""

    # Ensure urlparse can detect the netloc even if scheme is missing
    to_parse = u if "://" in u or u.startswith("//") else f"//{u}"
    parsed = urlparse(to_parse)

    host = parsed.hostname or ""
    # Normalize: remove any trailing dot from FQDNs
    return host.rstrip(".")

def _to_idna(host: str) -> str:
    try:
        return host.encode("idna").decode("ascii")
    except Exception:
        return host

def _normalize_hostname(host: str) -> str:
    if not host:
        return ""
    return _to_idna(host.strip().rstrip(".").lower())

def check_trusted_domain(url: str, trusted_domains: Iterable[str], allow_subdomains: bool = True) -> bool:
    """
    Check whether the URL's domain is trusted.

    Returns True if the domain extracted from the URL is either:
      - exactly in the trusted_domains set, or
      - a subdomain of any trusted domain (when allow_subdomains is True).
    Otherwise returns False.

    Notes:
      - Comparison is case-insensitive and normalizes Unicode domains using IDNA.
      - If trusted_domains is empty or None, returns False.
    """
    domain = extract_domain(url)
    if not domain:
        return False

    host = _normalize_hostname(domain)
    if not host:
        return False

    # Normalize the trusted domains list
    normalized_trusted = {
        _normalize_hostname(str(d)) for d in (trusted_domains or []) if str(d).strip()
    }
    if not normalized_trusted:
        return False

    if host in normalized_trusted:
        return True

    if allow_subdomains:
        for d in normalized_trusted:
            # Match subdomains like "sub.example.com" for trusted "example.com"
            if host.endswith("." + d):
                return True

    return False
