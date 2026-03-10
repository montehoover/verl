import re


def is_javascript_scheme(s):
    """
    Determine if a string contains a possibly malicious URL scheme.
    
    This function checks for potentially dangerous URL schemes like javascript:,
    vbscript:, etc., while allowing safe image data URLs.
    
    Args:
        s (str): The string to be checked for possibly malicious URL schemes.
        
    Returns:
        bool: True if the string contains a possibly malicious URL scheme,
              False otherwise (including for image data URLs which are safe).
    """
    # Pattern to match potentially malicious URL schemes
    # Matches: javascript:, jscript:, vbscript:, mocha:, livescript:, 
    # ecmascript:, or about: (with optional whitespace after scheme name)
    malicious_schemes_pattern = r'(?:javascript|jscript|vbscript|mocha|livescript|ecmascript|about)\s*:'
    
    # Pattern to match image data URLs (these are considered safe)
    # Matches: data:image/[format]; where format is one of the common image types
    image_data_pattern = r'data:image/(?:png|jpg|jpeg|gif|bmp|webp|svg\+xml|x-icon|vnd\.microsoft\.icon);'
    
    # Guard clause: Return early if it's a safe image data URL
    if re.search(image_data_pattern, s, re.IGNORECASE):
        return False
    
    # Check for malicious schemes and return the result directly
    return bool(re.search(malicious_schemes_pattern, s, re.IGNORECASE))
