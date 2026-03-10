import re

def has_script_scheme(url: str) -> bool:
    """
    Determine whether a given string contains a risky URL scheme.
    
    Args:
        url: str, the input string to be inspected for potentially malicious URL schemes
        
    Returns:
        bool, returns True if the string contains any dangerous URL schemes, 
        ignoring valid image data URLs; otherwise, False
    """
    # Convert to lowercase for case-insensitive matching
    url_lower = url.lower().strip()
    
    # List of dangerous URL schemes
    dangerous_schemes = [
        'javascript:', 'vbscript:', 'vbs:', 'mocha:', 'livescript:', 
        'ecmascript:', 'jscript:', 'behavior:', 'expression(',
        'mhtml:', 'x-javascript:', 'about:', 'chrome:', 'file:',
        'res:', 'ms-its:', 'mhtml:', 'opera:', 'safari-extension:'
    ]
    
    # Check for dangerous schemes
    for scheme in dangerous_schemes:
        if url_lower.startswith(scheme):
            return True
    
    # Special handling for data: URLs
    if url_lower.startswith('data:'):
        # Valid image data URLs pattern: data:image/<type>;...
        # We consider data URLs safe only if they explicitly specify an image MIME type
        image_data_pattern = re.compile(r'^data:\s*image/[^;,]+[;,]', re.IGNORECASE)
        if not image_data_pattern.match(url):
            # If it's a data URL but not a valid image data URL, it's risky
            return True
    
    return False
