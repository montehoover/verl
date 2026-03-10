import re

def verify_path_format(address: str) -> bool:
    """
    Verifies if a given string is a valid http or https path based on a regular expression pattern.

    Args:
        address: str, the input string to validate as a path.

    Returns:
        True if the path is formatted correctly, and False otherwise.
        Raises no exceptions.
    """
    # Regex to check for valid http or https URL.
    # This regex is a common pattern and covers most cases but might not be exhaustive for all edge cases of URLs.
    # It checks for:
    # - http:// or https:// protocol
    # - domain name (allowing for subdomains, alphanumeric characters, hyphens)
    # - optional port number
    # - optional path
    # - optional query string
    # - optional fragment identifier
    pattern = re.compile(
        r'^(https?://)'  # http:// or https://
        r'([A-Za-z0-9.-]+)'  # domain name or IP address
        r'(:\d+)?'  # optional port
        r'(/[^?#]*)?'  # optional path
        r'(\?[^#]*)?'  # optional query string
        r'(#.*)?$',  # optional fragment
        re.IGNORECASE)
    
    if re.fullmatch(pattern, address):
        return True
    return False
