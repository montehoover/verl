from urllib.parse import urlparse

APPROVED_DOMAINS = ['example.com', 'trusted.com']


def is_valid_domain(url: str) -> bool:
    """
    Return True if the URL's domain is in the approved list, otherwise False.
    Accepts bare domains (e.g., "example.com/path") by assuming http scheme.
    """
    if not isinstance(url, str) or not url.strip():
        return False

    candidate = url.strip()
    parsed = urlparse(candidate if '://' in candidate else f'http://{candidate}')
    host = (parsed.hostname or '').rstrip('.').lower()
    if not host:
        return False

    for approved in APPROVED_DOMAINS:
        normalized = approved.strip('.').lower()
        if host == normalized or host.endswith('.' + normalized):
            return True

    return False
