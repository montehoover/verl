import re

def detect_malicious_url_scheme(url_string: str) -> bool:
    """
    Determines whether a given string contains a risky URL scheme,
    excluding valid image data URLs.

    Args:
        url_string: The input string to be inspected.

    Returns:
        True if the string contains any dangerous URL schemes (ignoring
        valid image data URLs); otherwise, False.
    """
    # Regular expression to match common risky URL schemes.
    # Case-insensitive matching.
    # Allows for optional spaces/tabs/newlines after the scheme name and before the colon.
    malicious_scheme_pattern = re.compile(
        r"^(javascript|vbscript|data:(?!image\/(jpeg|png|gif|bmp|webp|svg\+xml))[^,]*):", 
        re.IGNORECASE | re.UNICODE
    )

    # Check for malicious schemes
    if malicious_scheme_pattern.search(url_string):
        return True
    
    return False
