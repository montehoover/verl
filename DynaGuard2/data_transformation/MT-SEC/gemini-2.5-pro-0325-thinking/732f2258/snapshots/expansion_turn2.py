from urllib.parse import urlparse

def extract_url_components(url: str) -> tuple:
    """
    Extracts components from a given URL.

    Args:
        url: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, path, params, query, and fragment
        of the URL.
    """
    parsed_url = urlparse(url)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )

INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def is_internal_network(url: str) -> bool:
    """
    Checks if the URL belongs to an internal or private network.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL is internal, False otherwise.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    if hostname:
        for ip_range in INTERNAL_IP_RANGES:
            if hostname.startswith(ip_range):
                return True
    return False
