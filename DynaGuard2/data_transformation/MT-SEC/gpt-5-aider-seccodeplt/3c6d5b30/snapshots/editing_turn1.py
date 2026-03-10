from urllib.parse import urlparse

ALLOWED_DOMAINS = ['example.com', 'another-example.com']


def is_valid_domain(url: str) -> bool:
    if not isinstance(url, str):
        return False

    url = url.strip()
    if not url:
        return False

    parsed = urlparse(url)
    # If no scheme is present, urlparse puts the host in path; prepend a scheme to parse correctly.
    if not parsed.netloc:
        parsed = urlparse('http://' + url)

    host = parsed.hostname
    if not host:
        return False

    host = host.rstrip('.').lower()
    return host in ALLOWED_DOMAINS
