import re
import logging


# Configure logger for this module
logger = logging.getLogger(__name__)


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
    
    # Guard clause: First check if the URL is a safe image data URL
    # These are explicitly allowed even though they start with 'data:'
    if safe_image_data_pattern.match(url_str):
        logger.debug(f"Safe image data URL detected: {url_str[:50]}...")
        return False
    
    # Guard clause: Check if the URL contains any dangerous scheme
    if not dangerous_scheme_pattern.match(url_str):
        return False
    
    # Log the detection of a dangerous URL scheme
    scheme_match = dangerous_scheme_pattern.match(url_str)
    if scheme_match:
        detected_scheme = scheme_match.group(1)
        logger.warning(
            f"Dangerous URL scheme detected: '{detected_scheme}' in URL: {url_str[:100]}..."
        )
    
    return True
