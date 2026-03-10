import re


def verify_path_format(address: str) -> bool:
    """
    Verify if a given string is a valid HTTP or HTTPS URL.
    
    This function uses a regular expression pattern to validate whether
    the input string is a properly formatted HTTP or HTTPS URL.
    
    Args:
        address (str): The input string to validate as a URL.
        
    Returns:
        bool: True if the address is a valid HTTP/HTTPS URL, False otherwise.
        
    Examples:
        >>> verify_path_format("https://example.com")
        True
        >>> verify_path_format("http://subdomain.example.com/path")
        True
        >>> verify_path_format("ftp://example.com")
        False
        >>> verify_path_format("not a url")
        False
    """
    # Regular expression pattern to match HTTP/HTTPS URLs
    # ^https?:// - Starts with http:// or https://
    # [^\s/$.?#] - First character after protocol must not be whitespace or special chars
    # . - Any character (ensures at least one character after protocol)
    # [^\s]* - Followed by any non-whitespace characters
    # $ - End of string
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    
    return bool(re.match(pattern, address))
