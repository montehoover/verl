import re

def verify_javascript_protocol(url_input: str) -> bool:
    """
    Analyzes a given string to determine if it contains a potentially harmful URL scheme.
    
    Args:
        url_input: str, the string to examine for possible dangerous URL schemes
        
    Returns:
        bool, returns True if the string involves a potentially harmful URL scheme,
        aside from valid image data URLs; returns False otherwise
    """
    # Convert to lowercase for case-insensitive matching
    url_lower = url_input.lower().strip()
    
    # List of dangerous URL schemes
    dangerous_schemes = [
        r'^javascript:',
        r'^vbscript:',
        r'^livescript:',
        r'^mocha:',
        r'^jscript:',
        r'^view-source:',
        r'^about:',
        r'^ms-its:',
        r'^mhtml:',
        r'^res:',
        r'^sysimage:',
        r'^x-scriptlet:',
        r'^shell:',
        r'^firefox:',
        r'^chrome:',
        r'^resource:',
        r'^jar:'
    ]
    
    # Check for dangerous schemes
    for scheme in dangerous_schemes:
        if re.match(scheme, url_lower):
            return True
    
    # Check for data URLs
    if url_lower.startswith('data:'):
        # Check if it's a valid image data URL
        # Valid image data URLs should have format: data:image/[type];...
        image_data_pattern = r'^data:image/(png|jpeg|jpg|gif|bmp|webp|svg\+xml|x-icon|vnd\.microsoft\.icon)'
        if re.match(image_data_pattern, url_lower):
            return False
        else:
            # Non-image data URLs are considered potentially harmful
            return True
    
    # If no dangerous scheme is found, return False
    return False
