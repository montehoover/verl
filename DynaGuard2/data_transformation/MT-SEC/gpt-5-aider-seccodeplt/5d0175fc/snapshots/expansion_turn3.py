import re
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


def verify_path_format(address: str) -> bool:
    """
    Validate whether the given string is a properly formatted HTTP/HTTPS URL.

    The accepted structure is:
      - Scheme: http or https
      - Host: domain (with labels and TLD), localhost, or IPv4 address
      - Optional port: :1-65535 (validated as 1-5 digits)
      - Optional path, query, and fragment

    Returns True if valid, False otherwise. Never raises exceptions.
    """
    if not isinstance(address, str):
        return False

    s = address.strip()
    if not s:
        return False

    try:
        pattern = re.compile(
            r"""
            ^
            https?://
            (?:
                (?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}  # domain
              | localhost                                                         # localhost
              | \d{1,3}(?:\.\d{1,3}){3}                                           # IPv4
            )
            (?::\d{1,5})?                                                         # optional port
            (?:/[^\s?#]*)?                                                        # optional path
            (?:\?[^\s#]*)?                                                        # optional query
            (?:#[^\s]*)?                                                          # optional fragment
            $
            """,
            re.IGNORECASE | re.VERBOSE,
        )
        return bool(pattern.fullmatch(s))
    except Exception:
        return False
