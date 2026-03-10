import re

def check_javascript_scheme(url_str: str) -> bool:
    """
    Determines whether a given string contains a potentially harmful URL scheme.
    
    Args:
        url_str: str, the string in question to be examined for potentially dangerous URL schemes
        
    Returns:
        bool, returns True if the input string contains a dangerous URL scheme (except safe image data URLs);
        returns False otherwise
    """
    # Pattern to match dangerous URL schemes
    dangerous_schemes = re.compile(
        r'^\s*(javascript|jscript|vbscript|vbs|about|wysiwyg|data|mhtml|file|res):', 
        re.IGNORECASE
    )
    
    # Pattern to match safe image data URLs
    safe_image_data = re.compile(
        r'^\s*data:\s*image/(gif|png|jpeg|jpg|webp|svg\+xml);', 
        re.IGNORECASE
    )
    
    # Check if it's a safe image data URL first
    if safe_image_data.match(url_str):
        return False
    
    # Check if it contains a dangerous scheme
    if dangerous_schemes.match(url_str):
        return True
    
    return False
