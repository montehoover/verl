from urllib.parse import urlparse

APPROVED_DOMAINS = {'example.com', 'test.com', 'myservice.com'}


def validate_url_domain(url: str) -> bool:
    """
    Returns True if the URL's hostname matches exactly one of the approved domains, else False.
    - Performs a simple exact-match against the hostname (not subdomains).
    - If the URL lacks a scheme, it attempts to parse by prefixing 'http://'.
    """
    if not isinstance(url, str) or not url:
        return False

    parsed = urlparse(url)
    if not parsed.netloc:
        parsed = urlparse(f"http://{url}")

    hostname = parsed.hostname
    if not hostname:
        return False

    hostname = hostname.lower().strip(".")
    return hostname in APPROVED_DOMAINS
