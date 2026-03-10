import re

def analyze_scheme(url: str) -> bool:
    """
    Analyzes a given string to determine if it contains a potentially harmful URL scheme.
    
    Args:
        url: str - the string to examine for possible dangerous URL schemes
        
    Returns:
        bool - True if the string involves a potentially harmful URL scheme, 
               aside from valid image data URLs; False otherwise
    """
    # List of dangerous URL schemes
    dangerous_schemes = [
        'javascript:', 'vbscript:', 'livescript:', 'mocha:', 
        'jscript:', 'view-source:', 'about:', 'ms-its:', 
        'mhtml:', 'vbs:', 'res:', 'x-javascript:'
    ]
    
    # Check if URL starts with any dangerous scheme
    url_lower = url.lower().strip()
    for scheme in dangerous_schemes:
        if url_lower.startswith(scheme):
            return True
    
    # Check for data: URLs
    if url_lower.startswith('data:'):
        # Image data URLs have MIME types like image/png, image/jpeg, image/gif, etc.
        # Pattern to match image data URLs
        image_data_pattern = r'^data:image/(png|jpeg|jpg|gif|svg\+xml|webp|bmp|ico|tiff);'
        
        # If it's a data URL but NOT an image data URL, it's potentially harmful
        if not re.match(image_data_pattern, url_lower):
            return True
    
    # If none of the above conditions are met, the URL scheme is safe
    return False
