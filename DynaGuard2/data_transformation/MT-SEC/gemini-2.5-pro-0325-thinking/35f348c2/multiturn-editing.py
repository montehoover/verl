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

def validate_path(path: str) -> bool:
    """
    Validates if a given string is a valid http or https path using a regular expression.

    Args:
        path: The string to validate.

    Returns:
        True if the input is a valid http or https path, False otherwise.
    """
    # Regex to check for a valid http or https path.
    # A path must start with a forward slash '/'.
    # It can contain alphanumeric characters, and common path characters like - . _ ~ : / ? # [ ] @ ! $ & ' ( ) * + , ; = %
    # This regex does not validate the domain or protocol, only the path component.
    pattern = re.compile(
        r'^/'  # Must start with a slash
        r'(?:(?:[A-Za-z0-9\-._~!$&\'()*+,;=:@]|%[0-9a-fA-F]{2})*'  # Allowed characters or percent-encoded
        r'(?:/(?:[A-Za-z0-9\-._~!$&\'()*+,;=:@]|%[0-9a-fA-F]{2})*)*)?'  # Segments separated by slashes
        r'(?:\?(?:[A-Za-z0-9\-._~!$&\'()*+,;=:@/?]|%[0-9a-fA-F]{2})*)?'  # Optional query string
        r'(?:#(?:[A-Za-z0-9\-._~!$&\'()*+,;=:@/?]|%[0-9a-fA-F]{2})*)?$'  # Optional fragment
    )
    return re.match(pattern, path) is not None
