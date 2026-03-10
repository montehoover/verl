import re

def analyze_scheme(url: str) -> bool:
    """
    Analyzes a given string to determine if it contains a potentially harmful URL scheme,
    such as javascript, vbscript, or similar, while ignoring image data URLs from the verification.

    Args:
        url: str, the string to examine for possible dangerous URL schemes.

    Returns:
        bool, returns True if the string involves a potentially harmful URL scheme,
        aside from valid image data URLs; returns False otherwise.
    
    Raises:
        The function doesn't raise specific exceptions. However, invalid inputs
        or malformed regex could lead to 're.error' exceptions.
    """
    # Convert URL to lowercase for case-insensitive matching
    lower_url = url.lower()

    # Pattern for explicitly harmful schemes like javascript: or vbscript:
    harmful_direct_schemes_pattern = r"^(javascript|vbscript):"

    if re.search(harmful_direct_schemes_pattern, lower_url):
        return True

    # Pattern for data: URLs
    data_scheme_pattern = r"^data:"
    # Pattern for safe image data: URLs (e.g., data:image/png, data:image/jpeg)
    image_data_scheme_pattern = r"^data:image/"

    if re.search(data_scheme_pattern, lower_url):
        # If it's a data: URL, check if it's NOT an image data URL
        if not re.search(image_data_scheme_pattern, lower_url):
            # This is a data: URL but not for an image, considered harmful
            return True
        else:
            # This is a data:image/... URL, which is considered safe/ignored
            return False
            
    # If none of the above conditions are met, the URL scheme is not considered harmful
    return False
