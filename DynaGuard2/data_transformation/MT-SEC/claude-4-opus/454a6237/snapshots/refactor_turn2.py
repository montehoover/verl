import re


def check_path_validity(path_string: str) -> bool:
    """Verify if a given string is a valid HTTP or HTTPS URL.
    
    This function uses a regular expression pattern to validate whether
    the input string is a properly formatted HTTP or HTTPS URL.
    
    Args:
        path_string: The input string to validate as a URL.
        
    Returns:
        True if the path is formatted correctly as an HTTP/HTTPS URL,
        False otherwise.
    """
    # Regular expression pattern explanation:
    # ^           - Start of string
    # https?      - Matches 'http' or 'https' (? makes 's' optional)
    # ://         - Matches the protocol separator
    # [^\s/$.?#]  - Matches any character except whitespace and special chars
    # .           - Matches any character (at least one character for domain)
    # [^\s]*      - Matches any non-whitespace character (rest of URL)
    # $           - End of string
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    
    # Check if the path_string matches the pattern
    if re.match(pattern, path_string):
        return True
    else:
        return False
