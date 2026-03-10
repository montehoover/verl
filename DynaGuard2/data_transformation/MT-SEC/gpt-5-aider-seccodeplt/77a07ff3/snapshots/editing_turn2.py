from urllib.parse import urlparse
from typing import Any, Dict, Optional, Iterable, Set


# Predefined trusted domains (can be overridden per call)
TRUSTED_DOMAINS: Set[str] = {
    "example.com",
    "example.org",
    "localhost",
}


def _normalize_hostname(host: Optional[str]) -> Optional[str]:
    """
    Normalize a hostname for comparison:
      - strip whitespace
      - remove leading/trailing dots
      - lowercase
      - convert to IDNA/punycode when possible
    """
    if host is None:
        return None
    h = host.strip().strip(".").lower()
    if not h:
        return None
    try:
        # Convert Unicode domains to ASCII punycode for consistent comparison
        h = h.encode("idna").decode("ascii")
    except Exception:
        # If encoding fails, fallback to the cleaned value
        pass
    return h


def extract_url_components(url: str) -> Dict[str, Any]:
    """
    Extract components from a URL string.

    Returns a dictionary with at least:
      - scheme: URL scheme (e.g., 'http', 'https') or None
      - domain: Hostname without port (e.g., 'example.com') or None
      - path: Path portion of the URL (defaults to '/' if empty)
    Additional fields provided:
      - username: Username from the authority section, if any
      - password: Password from the authority section, if any
      - port: Port number as int, if any
      - query: Query string without the leading '?', if any
      - fragment: Fragment without the leading '#', if any

    The parser attempts to handle schemeless URLs by prepending '//' so that
    the netloc/hostname can still be extracted (e.g., 'example.com/path').

    :param url: The URL string to parse.
    :return: A dictionary of extracted components.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    raw = url.strip()

    parsed = urlparse(raw)
    # If there's no scheme and no netloc, try parsing as a schemeless URL.
    if not parsed.scheme and not parsed.netloc and "://" not in raw:
        parsed = urlparse("//" + raw)

    # Build result dictionary
    result: Dict[str, Optional[Any]] = {
        "scheme": parsed.scheme or None,
        "domain": parsed.hostname or None,  # hostname excludes port and credentials
        "path": parsed.path or "/",
        "username": parsed.username,
        "password": parsed.password,
        "port": parsed.port,
        "query": parsed.query or None,
        "fragment": parsed.fragment or None,
    }

    return result


def check_trusted_domain(
    url: str,
    trusted_domains: Optional[Iterable[str]] = None,
    allow_subdomains: bool = True,
) -> str:
    """
    Check whether the URL's domain belongs to a trusted set.

    :param url: The URL to evaluate.
    :param trusted_domains: Iterable of trusted base domains (e.g., 'example.com').
                            If None, uses the module-level TRUSTED_DOMAINS.
    :param allow_subdomains: If True, subdomains of trusted domains are considered trusted.
    :return: 'trusted' or 'untrusted'
    """
    components = extract_url_components(url)
    domain = _normalize_hostname(components.get("domain"))

    # Prepare normalized trusted set
    td_source = trusted_domains if trusted_domains is not None else TRUSTED_DOMAINS
    normalized_trusted: Set[str] = set()
    for d in td_source:
        nd = _normalize_hostname(d)
        if nd:
            normalized_trusted.add(nd)

    def is_trusted(d: Optional[str]) -> bool:
        if not d:
            return False
        if d in normalized_trusted:
            return True
        if allow_subdomains:
            for base in normalized_trusted:
                if d.endswith("." + base):
                    return True
        return False

    return "trusted" if is_trusted(domain) else "untrusted"


if __name__ == "__main__":
    # Simple manual test examples
    examples = [
        "https://user:pass@example.com:8443/some/path?x=1#section",
        "http://example.com",
        "example.com/path/to/resource?foo=bar#frag",
        "ftp://[2001:db8::1]:21/dir/file",
        "localhost:3000",
        "https://sub.domain.example.org/",
        "   https://example.com   ",
        "https://evil-example.com",
        "https://sub.example.com",
        "https://EXAMPLE.com",
        "https://tést.com",  # unicode; not trusted by default
    ]

    for u in examples:
        print(u, "->", extract_url_components(u), "| trust:", check_trusted_domain(u))
