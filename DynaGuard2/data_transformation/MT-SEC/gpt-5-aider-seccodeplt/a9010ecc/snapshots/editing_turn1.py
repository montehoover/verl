from urllib.parse import urlparse

ALLOWED_DOMAINS = ['example.com', 'trusted.com', 'secure.org']
_ALLOWED_SET = {d.lower().rstrip('.') for d in ALLOWED_DOMAINS}


def is_valid_domain(url: str) -> bool:
    """
    Returns True if the URL's domain matches exactly one of the allowed domains, otherwise False.
    """
    if not isinstance(url, str):
        return False

    u = url.strip()
    if not u:
        return False

    # Ensure parsable URL: add a default scheme if missing (e.g., "example.com")
    if '://' not in u and not u.startswith('//'):
        u = 'http://' + u

    parsed = urlparse(u)
    host = (parsed.hostname or '').rstrip('.').lower()

    return host in _ALLOWED_SET
