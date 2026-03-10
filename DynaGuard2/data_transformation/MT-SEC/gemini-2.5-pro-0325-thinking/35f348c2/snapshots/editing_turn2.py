import re

def is_http_url(url: str) -> bool:
    """
    Checks if a given string starts with 'http://' or 'https://'.

    Args:
        url: The string to check.

    Returns:
        True if the url starts with 'http://' or 'https://', False otherwise.
    """
    return url.startswith('http://') or url.startswith('https://')

def validate_url(url: str) -> bool:
    """
    Validates a URL using a regular expression.

    Args:
        url: The string to validate.

    Returns:
        True if the url is a valid URL, False otherwise.
    """
    # Regex to check for a valid URL (simplified for common cases)
    # This regex checks for http/https, domain name, and optional path/query/fragment
    pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(pattern, url) is not None
