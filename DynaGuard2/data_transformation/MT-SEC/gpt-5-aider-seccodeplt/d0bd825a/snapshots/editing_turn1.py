from urllib.parse import urlparse

APPROVED_DOMAINS = {'example.com', 'test.com', 'myservice.com'}


def validate_url_domain(url: str) -> bool:
    """
    Return True if the URL's domain is in the approved list (including subdomains), else False.
    """
    if not isinstance(url, str):
        return False

    url = url.strip()
    if not url:
        return False

    # Ensure the URL has a scheme so urlparse can extract hostname reliably
    parsed = urlparse(url if "://" in url else f"http://{url}")

    host = parsed.hostname
    if not host:
        return False

    host = host.lower().rstrip(".")  # normalize and remove trailing dot if present

    for domain in APPROVED_DOMAINS:
        d = domain.lower().rstrip(".")
        if host == d or host.endswith("." + d):
            return True

    return False
