import re

def contains_dangerous_scheme(user_input: str) -> bool:
    """
    Determine whether a given string contains a risky URL scheme.
    
    Args:
        user_input: str, the input string to be inspected for potentially malicious URL schemes
        
    Returns:
        bool, returns True if the string contains any dangerous URL schemes, 
        ignoring valid image data URLs; otherwise, False
    """
    # Define dangerous URL schemes to check for
    dangerous_schemes = [
        'javascript:', 'vbscript:', 'livescript:', 'mocha:',
        'jscript:', 'view-source:', 'about:', 'chrome:',
        'chrome-extension:', 'data:text/html', 'data:application/'
    ]
    
    # Convert input to lowercase for case-insensitive matching
    input_lower = user_input.lower()
    
    # First check if it's a data URL
    if input_lower.startswith('data:'):
        # Check if it's a valid image data URL
        # Valid image data URLs start with data:image/
        image_data_pattern = r'^data:image/[^;]+;'
        if re.match(image_data_pattern, input_lower):
            # It's an image data URL, so it's safe
            return False
        # It's a data URL but not an image, check if it's dangerous
        # data:text/html and data:application/ are considered dangerous
        if input_lower.startswith('data:text/html') or input_lower.startswith('data:application/'):
            return True
        # Other data URLs (like the example data:;base64) are considered dangerous
        return True
    
    # Check for other dangerous schemes
    for scheme in dangerous_schemes:
        if scheme in input_lower:
            return True
    
    return False
