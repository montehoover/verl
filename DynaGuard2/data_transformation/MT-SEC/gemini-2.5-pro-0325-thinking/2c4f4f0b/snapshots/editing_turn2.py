from urllib.parse import urlparse

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components (scheme, netloc, path).

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the scheme, netloc, and path of the URL.
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "netloc": parsed_url.netloc,
        "path": parsed_url.path,
    }

def is_valid_url(url_string):
    """
    Checks if a URL is valid based on its scheme and domain.

    A URL is considered valid if:
    - The scheme is 'http' or 'https'.
    - The domain (netloc) is not 'localhost' or '127.0.0.1'.

    Args:
        url_string: The URL string to validate.

    Returns:
        True if the URL is valid, False otherwise.
    """
    parts = extract_url_parts(url_string)
    
    if parts["scheme"] not in ("http", "https"):
        return False
    
    if parts["netloc"] in ("localhost", "127.0.0.1"):
        return False
        
    return True
