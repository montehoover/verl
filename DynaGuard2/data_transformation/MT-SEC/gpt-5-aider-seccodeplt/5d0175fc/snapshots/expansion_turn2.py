from typing import Optional, Dict
from urllib.parse import urlparse, urlsplit


def extract_protocol(url: str) -> Optional[str]:
    """
    Extract the protocol/scheme from a URL string.

    Args:
        url: The URL as a string.

    Returns:
        The protocol (e.g., 'http', 'https') as a lowercase string,
        or None if no protocol is present.
    """
    if not isinstance(url, str):
        return None

    s = url.strip()
    if not s:
        return None

    parsed = urlparse(s)
    scheme = parsed.scheme
    if scheme:
        return scheme.lower()
    return None


def parse_url_components(url: str) -> Dict[str, Optional[str]]:
    """
    Parse a URL string into its components: protocol, domain, and path.
    - protocol: scheme of the URL (e.g., 'http', 'https'), or None if absent
    - domain: hostname (e.g., 'www.example.com'), without port/userinfo, or None if absent
    - path: everything after the domain, including path, query, and fragment (may be empty string)

    Args:
        url: The URL as a string.

    Returns:
        A dictionary with keys 'protocol', 'domain', and 'path'.
    """
    result: Dict[str, Optional[str]] = {"protocol": None, "domain": None, "path": ""}

    if not isinstance(url, str):
        return result

    s = url.strip()
    if not s:
        return result

    def build_path(p) -> str:
        path = p.path or ""
        if p.query:
            path += f"?{p.query}"
        if p.fragment:
            path += f"#{p.fragment}"
        return path

    # First attempt: parse as-is
    p = urlsplit(s)
    result["protocol"] = p.scheme.lower() if p.scheme else None
    result["domain"] = p.hostname if p.hostname else None
    result["path"] = build_path(p)

    # If no scheme and no hostname, try parsing with // prefix to capture bare domains
    if result["domain"] is None and result["protocol"] is None:
        p2 = urlsplit("//" + s)
        if p2.hostname:
            result["domain"] = p2.hostname
            result["path"] = build_path(p2)

    return result
