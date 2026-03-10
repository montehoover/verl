def is_http_protocol(url_string: str) -> bool:
    """
    Checks if a given string starts with 'http' or 'https'.

    Args:
        url_string: The string to check.

    Returns:
        True if the string starts with 'http' or 'https', False otherwise.
    """
    return url_string.startswith('http://') or url_string.startswith('https://')


from urllib.parse import urlparse

def extract_url_components(url_string: str) -> dict:
    """
    Extracts protocol, domain, and path from a URL string.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary with keys 'protocol', 'domain', and 'path'.
        Returns None for these values if the URL is malformed or the component is not present.
    """
    parsed_url = urlparse(url_string)
    protocol = parsed_url.scheme if parsed_url.scheme else None
    domain = parsed_url.netloc if parsed_url.netloc else None
    path = parsed_url.path if parsed_url.path else None

    # Ensure path starts with a '/' if it exists and is not empty,
    # and domain is present.
    if domain and path and not path.startswith('/'):
        path = '/' + path
    elif domain and not path: # if there's a domain but no path, path should be '/'
        path = '/'


    return {
        'protocol': protocol,
        'domain': domain,
        'path': path
    }
