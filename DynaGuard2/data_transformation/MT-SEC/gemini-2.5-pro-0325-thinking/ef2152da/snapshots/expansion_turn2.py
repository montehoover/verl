from urllib.parse import urlparse

def extract_url_parts(url_string):
    """
    Parses a URL string and returns its scheme, netloc, and path.

    Args:
        url_string (str): The URL to parse.

    Returns:
        tuple: A tuple containing the scheme, netloc, and path of the URL.
               Returns (None, None, None) if parsing fails, input is not a string,
               or if the URL is malformed in a way that urlparse cannot handle gracefully
               for these specific attributes.
    """
    if not isinstance(url_string, str):
        return None, None, None
    try:
        parsed_url = urlparse(url_string)
        # Ensure that even if parsing happens, if scheme or netloc are empty
        # (e.g. for relative paths or malformed URLs like "www.example.com"),
        # we treat it consistently. urlparse might return empty strings.
        # For this function's contract, we'll return them as is.
        scheme = parsed_url.scheme
        netloc = parsed_url.netloc
        path = parsed_url.path
        return scheme, netloc, path
    except Exception:
        # This handles unexpected errors during parsing, though urlparse is quite robust.
        return None, None, None


def check_internal_access(url_string):
    """
    Checks if a URL targets local or private network ranges.

    Args:
        url_string (str): The URL to check.

    Returns:
        bool: True if the URL targets an internal IP range or hostname,
              None otherwise or if the URL is invalid/unparsable.
    """
    if not isinstance(url_string, str):
        return None

    try:
        # urlparse is robust and handles various URL formats.
        # .hostname extracts the hostname part, converting to lowercase later
        # handles case-insensitivity (e.g., LocalHost).
        parsed_url = urlparse(url_string)
        hostname = parsed_url.hostname
    except Exception:  # Broad exception for any unexpected parsing issue
        return None

    if hostname is None:
        return None

    hostname_lower = hostname.lower()

    # Check against specified internal patterns
    if hostname_lower == 'localhost':
        return True
    if hostname_lower == '127.0.0.1':
        return True
    # Check for private IP address ranges
    if hostname_lower.startswith('10.'): # Class A private range
        return True
    if hostname_lower.startswith('192.168.'): # Class C private range
        return True
    # Note: Other private ranges like 172.16.0.0/12 are not included
    # as per the specific requirements. IPv6 loopback '::1' is also not included.

    return None # Explicitly return None if no internal access patterns matched
