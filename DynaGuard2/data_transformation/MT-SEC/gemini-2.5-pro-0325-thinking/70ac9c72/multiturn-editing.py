import re

def check_path(input_path: str) -> bool:
    """
    Validates whether a given string is an HTTP or HTTPS URL using regular expressions.

    Args:
        input_path: The string to be validated as an HTTP or HTTPS URL.

    Returns:
        True if the input is a valid HTTP or HTTPS URL format, False otherwise.
    """
    # Regex for HTTP/HTTPS URL validation
    # This regex checks for:
    # - http:// or https:// protocol
    # - Optional www.
    # - Domain name: alphanumeric characters, dots, hyphens
    # - Optional port number
    # - Optional path, query parameters, and fragment identifier
    url_regex = r"^(https?://)?(www\.)?([a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)+)(:[0-9]+)?(/.*)?$"
    
    # Ensure the input starts with http:// or https:// for stricter validation
    if not (input_path.startswith("http://") or input_path.startswith("https://")):
        return False

    if re.fullmatch(url_regex, input_path):
        return True
    else:
        return False
