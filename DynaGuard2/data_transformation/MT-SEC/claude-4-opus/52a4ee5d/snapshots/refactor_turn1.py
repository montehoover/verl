import re

def detect_js_scheme(url_string: str) -> bool:
    """
    Analyze a given string to determine if it contains a potentially harmful URL scheme.
    
    Args:
        url_string: The string to examine for possible dangerous URL schemes
        
    Returns:
        True if the string involves a potentially harmful URL scheme (aside from valid image data URLs),
        False otherwise
    """
    # Pattern to match potentially harmful URL schemes
    # Matches: javascript:, vbscript:, data: (except image data URLs), etc.
    harmful_scheme_pattern = r'^\s*(javascript|vbscript|data):\s*'
    
    # Pattern to match safe image data URLs
    image_data_pattern = r'^\s*data:\s*image/(png|jpeg|jpg|gif|webp|svg\+xml|bmp|ico|tiff);'
    
    # Check if the URL string matches a harmful scheme pattern
    if re.match(harmful_scheme_pattern, url_string, re.IGNORECASE):
        # If it's a data: URL, check if it's an image
        if re.match(r'^\s*data:', url_string, re.IGNORECASE):
            # If it's an image data URL, it's safe
            if re.match(image_data_pattern, url_string, re.IGNORECASE):
                return False
        # For javascript:, vbscript:, or non-image data: URLs
        return True
    
    return False
