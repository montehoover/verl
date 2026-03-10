import re


def _is_harmful_scheme(url_string: str) -> bool:
    """
    Check if the URL string starts with a potentially harmful scheme.
    
    Args:
        url_string: The string to check for harmful schemes
        
    Returns:
        True if the string starts with javascript:, vbscript:, or data:
    """
    harmful_scheme_pattern = r'^\s*(javascript|vbscript|data):\s*'
    return bool(re.match(harmful_scheme_pattern, url_string, re.IGNORECASE))


def _is_image_data_url(url_string: str) -> bool:
    """
    Check if the URL string is a valid image data URL.
    
    Args:
        url_string: The string to check for image data URL format
        
    Returns:
        True if the string is a data URL for an image format
    """
    image_data_pattern = r'^\s*data:\s*image/(png|jpeg|jpg|gif|webp|svg\+xml|bmp|ico|tiff);'
    return bool(re.match(image_data_pattern, url_string, re.IGNORECASE))


def detect_js_scheme(url_string: str) -> bool:
    """
    Analyze a given string to determine if it contains a potentially harmful URL scheme.
    
    Args:
        url_string: The string to examine for possible dangerous URL schemes
        
    Returns:
        True if the string involves a potentially harmful URL scheme (aside from valid image data URLs),
        False otherwise
    """
    if not _is_harmful_scheme(url_string):
        return False
    
    # If it's a data: URL, check if it's a safe image format
    if url_string.strip().lower().startswith('data:'):
        return not _is_image_data_url(url_string)
    
    # For javascript: or vbscript: schemes
    return True
