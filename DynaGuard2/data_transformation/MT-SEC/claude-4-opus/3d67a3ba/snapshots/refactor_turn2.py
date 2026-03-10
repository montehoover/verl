import re


def check_url_format(link: str) -> bool:
    """
    Verify if a given string is a valid URL based on a regular expression pattern.
    
    This function validates URLs that start with http:// or https:// and checks
    for proper domain structure, localhost, or IP addresses. It also handles
    optional port numbers and paths.
    
    Args:
        link (str): The input string to validate as a URL.
        
    Returns:
        bool: True if the URL is formatted correctly, False otherwise.
        
    Examples:
        >>> check_url_format("https://www.example.com")
        True
        >>> check_url_format("http://localhost:8080/path")
        True
        >>> check_url_format("not-a-url")
        False
    """
    # Define the regular expression pattern for URL validation
    url_pattern = re.compile(
        # Protocol: Must start with http:// or https://
        r'^https?://'
        # Domain name validation
        r'(?:'
        # Option 1: Standard domain name
        r'(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        # Option 2: localhost
        r'localhost|'
        # Option 3: IP address (IPv4)
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        r')'
        # Optional port number (e.g., :8080)
        r'(?::\d+)?'
        # Optional path and query string
        r'(?:/?|[/?]\S+)$',
        re.IGNORECASE  # Case-insensitive matching
    )
    
    # Type check: ensure input is a string
    if not isinstance(link, str):
        return False
    
    # Perform regex match and return result
    return bool(url_pattern.match(link))
