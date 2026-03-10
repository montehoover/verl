import re

# Regex to validate HTTP/HTTPS URLs.
# This regex is designed to be reasonably comprehensive for common URL structures.
_URL_VALIDATION_REGEX = re.compile(
    r'^https?://'  # Scheme (http or https)
    # Authority part:
    r'(?:'  # Start of non-capturing group for authority
    # Option 1: Domain name
    # It matches sequences like 'example.com', 'sub.example.co.uk'
    # A label is alphanumeric, possibly with hyphens, not starting/ending with a hyphen.
    # TLD is alphabetic (e.g., .com) or can include numbers/hyphens for IDNs (e.g., .xn--p1ai)
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,63}|[A-Z0-9-]{2,63}\.?))|'
    # Option 2: localhost
    r'localhost|'
    # Option 3: IP address (v4)
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    r')'  # End of non-capturing group for authority
    # Optional port number (e.g., :80, :8080)
    r'(?::\d+)?'
    # Optional path, query, and fragment part.
    # It allows an optional slash, or a slash/question mark followed by non-whitespace characters.
    # This covers cases like '/', '/path', '/path?query=value#fragment'.
    r'(?:/?|[/?]\S*)?$'
    , re.IGNORECASE  # Make the regex case-insensitive
)

def is_valid_path(site_path: str) -> bool:
    """
    Determines if a given string is a valid http or https path based on
    matching it against a regular expression.

    Args:
        site_path: str, the string to be evaluated as a potential valid path.

    Returns:
        The function returns True if the string is a valid path, otherwise it returns False.
    """
    # According to the problem description, site_path is guaranteed to be a string.
    # If site_path could be other types (e.g., None), a type check like:
    # if not isinstance(site_path, str):
    #     return False
    # would be advisable to prevent TypeErrors from fullmatch and adhere to "no exceptions".

    if _URL_VALIDATION_REGEX.fullmatch(site_path):
        return True
    else:
        return False
