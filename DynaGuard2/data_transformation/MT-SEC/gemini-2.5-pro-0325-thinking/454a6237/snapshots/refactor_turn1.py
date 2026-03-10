import re

def check_path_validity(path_string: str) -> bool:
    """
    Verifies if a given string is a valid http or https path based on a regular expression pattern.

    Args:
        path_string: The input string to validate as a path.

    Returns:
        True if the path is formatted correctly, False otherwise.
    """
    # Regex to match http or https URLs.
    # It checks for:
    # - http:// or https:// protocol
    # - A domain name (allowing alphanumeric characters, hyphens, and dots)
    # - A top-level domain (at least 2 alphabetic characters)
    # - An optional path, query parameters, or fragment
    pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if re.fullmatch(pattern, path_string):
        return True
    return False
