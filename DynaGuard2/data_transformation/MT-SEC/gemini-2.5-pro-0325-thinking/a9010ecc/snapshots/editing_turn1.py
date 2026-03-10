from urllib.parse import urlparse

ALLOWED_DOMAINS = ['example.com', 'trusted.com', 'secure.org']

def is_valid_domain(url: str) -> bool:
    """
    Checks if the domain of the given URL is in the list of allowed domains.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL's domain is allowed, False otherwise.
    """
    if not isinstance(url, str):
        return False  # Or raise TypeError, depending on desired behavior

    try:
        parsed_url = urlparse(url)
        domain = parsed_url.hostname
        if domain and domain in ALLOWED_DOMAINS:
            return True
    except Exception:
        # Handle potential parsing errors, though urlparse is generally robust
        return False
    return False
