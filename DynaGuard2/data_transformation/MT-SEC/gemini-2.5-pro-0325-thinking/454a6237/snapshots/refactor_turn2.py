import re

def check_path_validity(path_string: str) -> bool:
    """
    Verifies if a given string is a valid http or https path based on a regular expression pattern.

    Args:
        path_string: The input string to validate as a path.

    Returns:
        True if the path is formatted correctly, False otherwise.

    The regular expression pattern is defined as follows:
    - `^https?://`: Matches "http://" or "https://" at the beginning of the string.
    - `(?: ... )`: Non-capturing group for the domain part.
        - `(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+`: Matches domain names.
          It allows for subdomains. Each part of the domain consists of
          alphanumeric characters and hyphens, but cannot start or end with a hyphen.
          The length of each part is between 1 and 63 characters.
        - `(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)`: Matches top-level domains (TLDs)
          like ".com", ".org", or country-code TLDs, or newer generic TLDs.
        - `|localhost`: Alternatively, matches "localhost".
        - `|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}`: Alternatively, matches an IP address
          (e.g., "192.168.1.1").
    - `(?::\d+)?`: Optionally matches a port number (e.g., ":8080").
    - `(?:/?|[/?]\S+)`: Optionally matches a path, query parameters, or fragment.
      It can be a single slash, or a slash followed by non-whitespace characters.
    - `$`: Matches the end of the string.
    - `re.IGNORECASE`: Makes the matching case-insensitive.
    """
    # Regex to match http or https URLs.
    pattern = re.compile(
        r'^https?://'  # Protocol: http:// or https://
        r'(?:'  # Start of non-capturing group for domain/IP
        # Domain name:
        # e.g., example.com, sub.example.co.uk
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'
        r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?))|'
        # localhost:
        r'localhost|'
        # IP address v4:
        # e.g., 127.0.0.1
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        r')'  # End of non-capturing group for domain/IP
        r'(?::\d+)?'  # Optional port number (e.g., :8080)
        r'(?:/?|[/?]\S+)?'  # Optional path, query, or fragment
        # (e.g., /, /path, /path?query=value, /path#fragment)
        # Allows an empty path (just domain), a single '/', or / followed by characters.
        # Using '?' at the end of this group to make the entire path part optional.
        r'$',  # End of string
        re.IGNORECASE  # Case-insensitive matching
    )

    # Use re.fullmatch to ensure the entire string matches the pattern.
    if re.fullmatch(pattern, path_string):
        return True
    return False
