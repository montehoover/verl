import re


def check_javascript_scheme(url_str: str) -> bool:
    """
    Determines whether a given string contains a potentially harmful URL scheme.
    
    Examines the input string for dangerous URL schemes like javascript, vbscript,
    etc., while allowing safe image data URLs (data:image/*).
    
    Args:
        url_str: str, the string in question to be examined for potentially 
                 dangerous URL schemes
        
    Returns:
        bool: True if the input string contains a dangerous URL scheme 
              (except safe image data URLs); False otherwise
              
    Raises:
        Although no specific exceptions are raised within this function,
        incorrectly formatted inputs or regex patterns could raise 
        're.error' exceptions.
    """
    # Define pattern for dangerous URL schemes that should be blocked
    dangerous_scheme_pattern = re.compile(
        r'^\s*(javascript|jscript|vbscript|vbs|about|wysiwyg|data|mhtml|file|res):', 
        re.IGNORECASE
    )
    
    # Define pattern for safe image data URLs that should be allowed
    safe_image_data_pattern = re.compile(
        r'^\s*data:\s*image/(gif|png|jpeg|jpg|webp|svg\+xml);', 
        re.IGNORECASE
    )
    
    # First check if the URL is a safe image data URL
    # These are explicitly allowed even though they start with 'data:'
    if safe_image_data_pattern.match(url_str):
        return False
    
    # Check if the URL contains any dangerous scheme
    if dangerous_scheme_pattern.match(url_str):
        return True
    
    # If neither condition is met, the URL is considered safe
    return False
