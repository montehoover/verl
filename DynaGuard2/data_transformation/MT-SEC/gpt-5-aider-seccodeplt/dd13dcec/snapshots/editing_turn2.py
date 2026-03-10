from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlparse


def extract_url_parts(url: str) -> Dict[str, Any]:
    """
    Parse a URL string and return its components as a dictionary.

    Returns keys:
      - scheme: str
      - username: Optional[str]
      - password: Optional[str]
      - hostname: Optional[str]
      - port: Optional[int]
      - path: str
      - params: str
      - query: str
      - fragment: str
      - netloc: str
    """
    parsed = urlparse(url)

    return {
        "scheme": parsed.scheme,
        "username": parsed.username,
        "password": parsed.password,
        "hostname": parsed.hostname,
        "port": parsed.port,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
        "netloc": parsed.netloc,
    }


def is_trusted_domain(url: str, trusted_domains: Sequence[str], include_subdomains: bool = True) -> bool:
    """
    Check whether the URL's hostname is in the list of trusted domains.

    Args:
        url: The URL to check.
        trusted_domains: A sequence of domain names (e.g., "example.com"). IP addresses are allowed.
        include_subdomains: If True, subdomains of trusted domains are also considered trusted.
            For example, with include_subdomains=True and trusted_domains=["example.com"],
            both "example.com" and "api.example.com" would be trusted.

    Returns:
        True if the URL's hostname matches any of the trusted domains (respecting include_subdomains), else False.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return False

    # Normalize hostname: lower-case and strip a possible trailing dot.
    host = host.rstrip(".").lower()

    # Normalize trusted domains similarly.
    normalized_trusted = [d.rstrip(".").lower() for d in trusted_domains if d]

    for domain in normalized_trusted:
        if not domain:
            continue

        if include_subdomains:
            # Exact match
            if host == domain:
                return True
            # Subdomain match: ensure we only match on label boundary using a dot.
            if host.endswith("." + domain):
                return True
        else:
            if host == domain:
                return True

    return False
