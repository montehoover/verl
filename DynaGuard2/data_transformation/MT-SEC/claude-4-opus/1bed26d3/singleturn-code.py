import re

def check_javascript_scheme(url_str: str) -> bool:
    """
    Determines whether a given string contains a potentially harmful URL scheme.
    
    Args:
        url_str: The string to be examined for potentially dangerous URL schemes
        
    Returns:
        True if the input string contains a dangerous URL scheme (except safe image data URLs);
        False otherwise
    """
    # List of dangerous URL schemes
    dangerous_schemes = [
        'javascript:', 'vbscript:', 'data:', 'about:', 'chrome:', 
        'chrome-extension:', 'ms-its:', 'mhtml:', 'file:', 'jar:', 
        'resource:', 'x-icon:', 'moz-icon:', 'livescript:', 'mocha:',
        'xss:', 'view-source:', 'res:', 'ie.http:', 'sysimage:', 
        'shell:', 'x-scriptlet:', 'jsmail:', 'opera:', 'firefoxurl:',
        'ms-help:', 'diskcopy:', 'mk:', 'mms:', 'tn3270:', 'acrobat:',
        'telnet:', 'mailserver:', 'cve:', 'ms-itss:', 'firefox:', 
        'chrome-devtools:', 'ms-windows-store:', 'read:', 'ms-browser-extension:'
    ]
    
    # Convert to lowercase for case-insensitive comparison
    url_lower = url_str.lower().strip()
    
    # Check if string starts with any dangerous scheme
    for scheme in dangerous_schemes:
        if url_lower.startswith(scheme):
            # Special handling for data: URLs
            if scheme == 'data:':
                # Check if it's a safe image data URL
                # Safe image MIME types: image/png, image/jpeg, image/jpg, image/gif, image/svg+xml, etc.
                image_pattern = re.compile(r'^data:image/[a-zA-Z0-9+.-]+[;,]', re.IGNORECASE)
                if image_pattern.match(url_str):
                    return False  # Safe image data URL
                else:
                    return True  # Dangerous data URL (non-image or no MIME type)
            else:
                return True  # Other dangerous schemes
    
    return False  # No dangerous scheme found
