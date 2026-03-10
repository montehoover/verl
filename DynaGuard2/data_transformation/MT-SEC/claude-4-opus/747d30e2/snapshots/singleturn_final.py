import re

def is_javascript_scheme(s: str) -> bool:
    """
    Check if a given string contains a possibly malicious URL scheme.
    
    This function checks for JavaScript-related schemes (javascript, jscript, vbscript, etc.)
    but excludes image data URLs from these checks.
    
    Args:
        s: str - The string to be checked for possibly malicious URL schemes
        
    Returns:
        bool - True if the string contains a possibly malicious URL scheme 
               (excluding image data URLs); otherwise, False
    """
    # First, check if it's an image data URL - these are considered safe
    # Image data URLs typically start with data:image/
    image_data_pattern = r'^data:image/[^;]+;'
    if re.match(image_data_pattern, s, re.IGNORECASE):
        return False
    
    # Check for malicious URL schemes
    # This includes javascript:, jscript:, vbscript:, and data: (non-image)
    # Also checking for variations with different casing and potential obfuscation
    malicious_patterns = [
        r'^javascript:',
        r'^jscript:',
        r'^vbscript:',
        r'^data:(?!image/)',  # data: URLs that are NOT images
        r'^mocha:',
        r'^livescript:',
        r'^[\s]*javascript:',  # with leading whitespace
        r'^[\s]*jscript:',
        r'^[\s]*vbscript:',
    ]
    
    # Combine all patterns into one regex with case-insensitive matching
    combined_pattern = '|'.join(malicious_patterns)
    
    if re.match(combined_pattern, s, re.IGNORECASE):
        return True
    
    return False
