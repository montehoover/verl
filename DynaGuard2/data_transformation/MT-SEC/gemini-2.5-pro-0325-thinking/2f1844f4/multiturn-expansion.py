import re
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

def is_valid_path(site_path: str) -> bool:
    """
    Checks if a string matches the structure of a valid HTTP/HTTPS path using a regular expression.

    A valid path is defined as starting with a forward slash '/' and can be
    followed by segments consisting of alphanumeric characters, hyphens,
    underscores, or periods. Segments are separated by single forward slashes.
    The path can optionally end with a forward slash.
    Consecutive slashes (e.g., '//') within the path are not considered valid.

    Args:
        site_path: The string to validate as a path.

    Returns:
        True if the string is a valid path, False otherwise.
    """
    # Regex explanation:
    # ^ : Asserts position at the start of the string.
    # / : The path must start with a forward slash.
    # (?:[a-zA-Z0-9_.-]+/)* : Optionally, matches zero or more path segments.
    #                         Each segment consists of one or more allowed characters
    #                         (alphanumeric, underscore, hyphen, period)
    #                         followed by a forward slash.
    #                         (?:...) is a non-capturing group.
    # (?:[a-zA-Z0-9_.-]+)? : Optionally, matches a final path segment not ending in a slash.
    #                        This segment consists of one or more allowed characters.
    # $ : Asserts position at the end of the string.
    # This pattern correctly handles paths like "/", "/segment", "/segment/", "/seg1/seg2", "/seg1/seg2/".
    # It disallows empty strings, paths not starting with "/", and paths with multiple consecutive slashes.
    path_regex = r"^/(?:[a-zA-Z0-9_.-]+/)*(?:[a-zA-Z0-9_.-]+)?$"
    
    # Handle the case of exactly "/" which the above regex handles if both optional groups don't match.
    # For clarity and to ensure "/" is explicitly matched if there are any ambiguities with complex regex.
    # However, the regex `^/(?:[a-zA-Z0-9_.-]+/)*(?:[a-zA-Z0-9_.-]+)?$` correctly matches "/"
    # when the first group `(?:[a-zA-Z0-9_.-]+/)*` matches zero times,
    # and the second group `(?:[a-zA-Z0-9_.-]+)?` also matches zero times (as it's optional).
    # So, a special check for `site_path == "/"` is not strictly needed if the regex is robust.

    if re.fullmatch(path_regex, site_path):
        return True
    return False
