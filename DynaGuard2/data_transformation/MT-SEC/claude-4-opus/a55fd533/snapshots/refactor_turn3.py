import re
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)

def _normalize_input(user_input: str) -> str:
    """Normalize the input string for consistent comparison."""
    normalized = user_input.lower().strip()
    logger.debug(f"Normalized input: '{user_input}' -> '{normalized}'")
    return normalized

def _is_safe_image_data_url(normalized_input: str) -> bool:
    """Check if the input is a safe image data URL."""
    safe_image_pattern = r'^data:image/(png|jpg|jpeg|gif|bmp|webp|ico);base64,'
    is_safe = bool(re.match(safe_image_pattern, normalized_input))
    if is_safe:
        logger.debug(f"Input identified as safe image data URL: '{normalized_input[:50]}...'")
    return is_safe

def _contains_any_dangerous_scheme(normalized_input: str) -> bool:
    """Check if the input contains any dangerous URL scheme."""
    dangerous_schemes = [
        'javascript:', 'vbscript:', 'data:text/html', 'data:application/javascript',
        'data:application/x-javascript', 'data:application/ecmascript', 
        'data:application/x-ecmascript', 'data:text/javascript',
        'data:text/ecmascript', 'data:text/x-javascript', 'data:text/x-ecmascript',
        'data:text/vbscript', 'data:text/x-vbscript', 'livescript:', 'mocha:',
        'jscript:', 'behavior:', 'chrome:', 'chrome-extension:', 'mhtml:',
        'data:image/svg+xml', 'data:text/xml'
    ]
    
    for scheme in dangerous_schemes:
        if scheme in normalized_input:
            logger.debug(f"Found dangerous scheme '{scheme}' in input")
            return True
    
    return False

def contains_dangerous_scheme(user_input: str) -> bool:
    """
    Determine whether a given string contains a risky URL scheme.
    
    Args:
        user_input: str, the input string to be inspected for potentially malicious URL schemes.
        
    Returns:
        bool, returns True if the string contains any dangerous URL schemes, ignoring valid image data URLs; otherwise, False.
    """
    logger.info(f"Checking input for dangerous schemes: '{user_input[:100]}{'...' if len(user_input) > 100 else ''}'")
    
    # Pipeline pattern: normalize -> check safe patterns -> check dangerous patterns
    normalized = _normalize_input(user_input)
    
    if _is_safe_image_data_url(normalized):
        logger.info(f"Input is safe (valid image data URL): '{user_input[:50]}...'")
        return False
    
    is_dangerous = _contains_any_dangerous_scheme(normalized)
    
    if is_dangerous:
        logger.warning(f"Dangerous scheme detected in input: '{user_input[:100]}{'...' if len(user_input) > 100 else ''}'")
    else:
        logger.info(f"No dangerous schemes found in input: '{user_input[:50]}...'")
    
    return is_dangerous
