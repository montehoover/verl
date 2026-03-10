from urllib.parse import urlparse

def extract_domain(url_string: str) -> str:
    """
    Extracts the domain from a URL string.

    Args:
        url_string: The URL string to parse.

    Returns:
        The domain part of the URL.
    """
    parsed_url = urlparse(url_string)
    return parsed_url.netloc
