import re

def is_valid_string(input_string: str) -> bool:
    """
    Checks if a given string is non-empty and doesn't contain any spaces.

    Args:
        input_string: The string to validate.

    Returns:
        True if the string is valid, False otherwise.
    """
    if not input_string:  # Check if the string is empty
        return False
    if ' ' in input_string:  # Check if the string contains any spaces
        return False
    return True

def check_url_format(link: str) -> bool:
    """
    Checks if a given string is a valid URL using a regular expression.

    Args:
        link: The string to validate as a URL.

    Returns:
        True if the URL is formatted correctly, False otherwise.
    """
    # Regex to check for a valid URL. This regex is a common one for basic URL validation.
    # It checks for http/https, optional www, domain name, and optional path/query parameters.
    url_pattern = re.compile(
        r'^(https?://)?'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if re.fullmatch(url_pattern, link):
        return True
    return False
