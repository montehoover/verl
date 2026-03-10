from urllib.parse import urlparse

def extract_url_components(url):
    """
    Extract components from a URL.

    Args:
        url (str): The URL to parse.

    Returns:
        tuple: A tuple containing (scheme, netloc, path, params, query, fragment).
    """
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment
