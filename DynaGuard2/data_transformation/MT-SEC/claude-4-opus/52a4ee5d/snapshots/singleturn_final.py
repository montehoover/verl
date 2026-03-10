import re

def detect_js_scheme(url_string: str) -> bool:
    """
    Analyzes a given string to determine if it contains a potentially harmful URL scheme.
    
    Args:
        url_string: str, the string to examine for possible dangerous URL schemes
        
    Returns:
        bool, returns True if the string involves a potentially harmful URL scheme,
        aside from valid image data URLs; returns False otherwise
    """
    # Define potentially harmful URL schemes
    harmful_schemes = [
        r'^\s*javascript\s*:',
        r'^\s*vbscript\s*:',
        r'^\s*livescript\s*:',
        r'^\s*mocha\s*:',
        r'^\s*jscript\s*:',
        r'^\s*view-source\s*:',
        r'^\s*about\s*:',
        r'^\s*ms-its\s*:',
        r'^\s*mhtml\s*:',
        r'^\s*x-javascript\s*:',
        r'^\s*xss\s*:',
    ]
    
    # Check for harmful schemes
    for pattern in harmful_schemes:
        if re.search(pattern, url_string, re.IGNORECASE):
            return True
    
    # Check for data URLs that are NOT image data URLs
    data_url_pattern = r'^\s*data\s*:'
    if re.search(data_url_pattern, url_string, re.IGNORECASE):
        # Check if it's a valid image data URL
        image_data_pattern = r'^\s*data\s*:\s*image\s*/\s*(png|jpeg|jpg|gif|svg\+xml|webp|bmp|ico|tiff)\s*(;|,)'
        if not re.search(image_data_pattern, url_string, re.IGNORECASE):
            # It's a data URL but not an image, so it's potentially harmful
            return True
    
    return False
