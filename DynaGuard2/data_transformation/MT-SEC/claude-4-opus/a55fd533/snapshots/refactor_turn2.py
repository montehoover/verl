import re

def _normalize_input(user_input: str) -> str:
    """Normalize the input string for consistent comparison."""
    return user_input.lower().strip()

def _is_safe_image_data_url(normalized_input: str) -> bool:
    """Check if the input is a safe image data URL."""
    safe_image_pattern = r'^data:image/(png|jpg|jpeg|gif|bmp|webp|ico);base64,'
    return bool(re.match(safe_image_pattern, normalized_input))

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
    
    return any(scheme in normalized_input for scheme in dangerous_schemes)

def contains_dangerous_scheme(user_input: str) -> bool:
    """
    Determine whether a given string contains a risky URL scheme.
    
    Args:
        user_input: str, the input string to be inspected for potentially malicious URL schemes.
        
    Returns:
        bool, returns True if the string contains any dangerous URL schemes, ignoring valid image data URLs; otherwise, False.
    """
    # Pipeline pattern: normalize -> check safe patterns -> check dangerous patterns
    normalized = _normalize_input(user_input)
    
    if _is_safe_image_data_url(normalized):
        return False
    
    return _contains_any_dangerous_scheme(normalized)
