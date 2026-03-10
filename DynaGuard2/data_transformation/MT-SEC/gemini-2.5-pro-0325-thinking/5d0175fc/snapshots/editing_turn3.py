import re

def verify_path_format(address: str) -> bool:
    """
    Checks if a given string is a valid HTTP or HTTPS path using a regular expression.

    Args:
        address: The string to check.

    Returns:
        True if the path is formatted correctly, False otherwise.
    """
    if not isinstance(address, str):
        return False
    
    # Regex for HTTP/HTTPS path validation.
    # It expects a path starting with '/' and can include alphanumeric characters,
    # hyphens, underscores, periods, and forward slashes.
    # It does not validate the domain or scheme, only the path component.
    # Example valid paths: "/path/to/resource", "/index.html", "/users/123/profile"
    # Example invalid paths: "path/without/leading/slash", "/path with spaces"
    regex = re.compile(
        r'^/'  # Must start with a forward slash
        r'(?:[a-zA-Z0-9\-_.~!$&\'()*+,;=:@%]|'  # Allowed characters in a path segment
        r'/[a-zA-Z0-9\-_.~!$&\'()*+,;=:@%]*)*'  # Further path segments
        r'(?:\?[a-zA-Z0-9\-_.~!$&\'()*+,;=:@%/?]*)?'  # Optional query string
        r'(?:#[a-zA-Z0-9\-_.~!$&\'()*+,;=:@%/?]*)?$'  # Optional fragment identifier
    )
    
    try:
        return re.match(regex, address) is not None
    except Exception:
        # Safeguard against unexpected errors, ensuring no exceptions are raised.
        return False
