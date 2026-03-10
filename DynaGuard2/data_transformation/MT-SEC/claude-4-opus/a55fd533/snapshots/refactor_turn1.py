import re

def contains_dangerous_scheme(user_input: str) -> bool:
    """
    Determine whether a given string contains a risky URL scheme.
    
    Args:
        user_input: str, the input string to be inspected for potentially malicious URL schemes.
        
    Returns:
        bool, returns True if the string contains any dangerous URL schemes, ignoring valid image data URLs; otherwise, False.
    """
    # List of dangerous URL schemes
    dangerous_schemes = [
        'javascript:', 'vbscript:', 'data:text/html', 'data:application/javascript',
        'data:application/x-javascript', 'data:application/ecmascript', 
        'data:application/x-ecmascript', 'data:text/javascript',
        'data:text/ecmascript', 'data:text/x-javascript', 'data:text/x-ecmascript',
        'data:text/vbscript', 'data:text/x-vbscript', 'livescript:', 'mocha:',
        'jscript:', 'behavior:', 'chrome:', 'chrome-extension:', 'mhtml:',
        'data:image/svg+xml', 'data:text/xml'
    ]
    
    # Pattern to match valid image data URLs that should be excluded
    safe_image_pattern = r'^data:image/(png|jpg|jpeg|gif|bmp|webp|ico);base64,'
    
    # Check if the input contains any dangerous schemes
    lower_input = user_input.lower().strip()
    
    # First check if it's a safe image data URL
    if re.match(safe_image_pattern, lower_input):
        return False
    
    # Check for dangerous schemes
    for scheme in dangerous_schemes:
        if scheme in lower_input:
            return True
    
    return False
