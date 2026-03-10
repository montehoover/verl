from urllib.parse import urlparse
from typing import Iterable, Optional


# Predefined list of allowed domains; modify as needed by your application.
ALLOWED_DOMAINS = {"example.com"}


def extract_url_components(url: str) -> dict:
    """
    Parse a URL string into its components.

    Returns a dictionary with keys:
      - scheme: URL scheme (e.g., 'http', 'https') or None
      - domain: hostname portion without credentials or port (e.g., 'example.com') or None
      - port: integer port if specified, else None
      - path: path component (e.g., '/a/b'), empty string if not present
      - params: parameters for the last path element (rarely used), empty string if not present
      - query: query string without the leading '?', empty string if not present
      - fragment: fragment without the leading '#', empty string if not present
      - username: username from credentials if present, else None
      - password: password from credentials if present, else None
    """
    parsed = urlparse(url)

    # If there's no scheme and no netloc, but the input likely represents a domain,
    # re-parse with '//' prefix to treat the first token as the netloc.
    if not parsed.scheme and not parsed.netloc:
        first_segment = url.split('/', 1)[0]
        looks_like_host = (
            bool(first_segment)
            and not url.startswith(('/', '?', '#'))
            and (
                '.' in first_segment
                or first_segment.startswith('localhost')
                or first_segment.startswith('[')  # IPv6 literal
            )
        )
        if looks_like_host:
            parsed = urlparse('//' + url)

    return {
        'scheme': parsed.scheme or None,
        'domain': parsed.hostname,
        'port': parsed.port,
        'path': parsed.path or '',
        'params': parsed.params or '',
        'query': parsed.query or '',
        'fragment': parsed.fragment or '',
        'username': parsed.username,
        'password': parsed.password,
    }


def _is_domain_allowed(domain: Optional[str], allowed_domains: Iterable[str]) -> bool:
    if not domain:
        return False
    host = domain.rstrip('.').lower()
    for allowed in allowed_domains:
        candidate = str(allowed).rstrip('.').lower()
        if not candidate:
            continue
        if host == candidate or host.endswith('.' + candidate):
            return True
    return False


def check_domain_approval(url: str, allowed_domains: Optional[Iterable[str]] = None) -> str:
    """
    Check whether the URL's domain is approved based on a predefined allowlist.

    - url: URL string to evaluate
    - allowed_domains: optional iterable of allowed domain suffixes; if None, uses ALLOWED_DOMAINS

    Returns:
      'approved' if the domain matches any allowed domain (exact or subdomain), else 'not approved'.
    """
    components = extract_url_components(url)
    allowlist = allowed_domains if allowed_domains is not None else ALLOWED_DOMAINS
    return 'approved' if _is_domain_allowed(components.get('domain'), allowlist) else 'not approved'
