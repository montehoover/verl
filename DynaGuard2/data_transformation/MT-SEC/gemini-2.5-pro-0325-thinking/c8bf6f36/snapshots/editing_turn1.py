from urllib.parse import urlparse

def extract_domain(url_string):
    """
    Extracts the domain name from a URL string.

    Args:
        url_string: The URL string.

    Returns:
        The domain name as a string.
    """
    parsed_url = urlparse(url_string)
    return parsed_url.netloc
