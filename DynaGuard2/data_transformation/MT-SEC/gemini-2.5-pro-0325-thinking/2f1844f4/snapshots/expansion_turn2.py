from urllib.parse import urlparse

def is_http_protocol(url: str) -> bool:
    """
    Checks if a URL starts with 'http' or 'https'.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL starts with 'http' or 'https', False otherwise.
    """
    return url.startswith('http://') or url.startswith('https://')

def extract_url_components(url: str) -> dict:
    """
    Extracts protocol, domain, and path from a URL string.

    Args:
        url: The URL string to parse.

    Returns:
        A dictionary with keys 'protocol', 'domain', and 'path'.
        - protocol: The section of the URL before '://'
        - domain: The section of the URL between '://' and the next '/'
        - path: The trailing content after the domain
    """
    parsed_url = urlparse(url)
    return {
        'protocol': parsed_url.scheme,
        'domain': parsed_url.netloc,
        'path': parsed_url.path
    }
