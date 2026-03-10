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
    # This regex is a common pattern and covers most cases but might not be
    # exhaustive for all edge cases of URLs.
    pattern = re.compile(
        r'^(https?://)'          # Scheme: http:// or https://
        r'([A-Za-z0-9.-]+)'      # Authority: domain name or IP address.
                                 # Allows for alphanumeric characters, dots, and hyphens.
        r'(:\d+)?'               # Port: optional, starts with a colon, followed by digits.
        r'(/[^?#]*)?'            # Path: optional, starts with a slash, can contain any characters
                                 # except '?' or '#'.
        r'(\?[^#]*)?'            # Query: optional, starts with '?', can contain any characters
                                 # except '#'.
        r'(#.*)?$',              # Fragment: optional, starts with '#', can contain any characters.
                                 # '$' asserts position at the end of the string.
        re.IGNORECASE            # Makes the regex case-insensitive.
    )

    # Attempt to match the entire string against the compiled pattern.
    # re.fullmatch() returns a match object if the whole string matches, otherwise None.
    return bool(re.fullmatch(pattern, address))
