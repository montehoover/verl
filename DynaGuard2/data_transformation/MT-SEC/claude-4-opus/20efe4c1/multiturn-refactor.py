import re


def verify_javascript_protocol(url_input: str) -> bool:
    """
    Analyzes a given string to determine if it contains a potentially harmful URL scheme.
    
    Checks for dangerous URL schemes like javascript:, vbscript:, and others that could
    be used for XSS attacks or other security vulnerabilities. Image data URLs are
    considered safe and are explicitly allowed.
    
    Args:
        url_input: str, the string to examine for possible dangerous URL schemes
        
    Returns:
        bool: returns True if the string involves a potentially harmful URL scheme,
              aside from valid image data URLs; returns False otherwise
    """
    # Guard clause: empty strings are safe
    if not url_input:
        return False
    
    # Normalize input by removing leading/trailing whitespace
    cleaned_url = url_input.strip()
    
    # Guard clause: image data URLs are explicitly allowed
    if re.match(r'^data:image/', cleaned_url, re.IGNORECASE):
        return False
    
    # Comprehensive pattern matching dangerous URL schemes
    # Includes: javascript, vbscript, data (non-image), and various
    # browser-specific or potentially harmful protocols
    dangerous_schemes = (
        r'^(javascript|vbscript|data:(?!image/)[^;]+|about|'
        r'chrome-extension|moz-extension|ms-browser-extension|'
        r'opera|res|resource|moz-icon|android-app|app|blob|'
        r'content|intent|moz-safe-about|moz-extension|sftp|ssh|'
        r'ftp|ftps|webdav|webdavs|dav|davs|afp|nfs|smb|'
        r'dav-noredirect|jar|xmpp|xmpp-bosh|sms|tel|wtai|'
        r'market|gopher|mms|nfs|nntp|telnet|imap|rtsp|tag|'
        r'sftp|ssh|bitcoin|magnet|aim|apt|attachment|cid|data|'
        r'dav|dns|fax|file|finger|gopher|im|imap|ldap|mailto|'
        r'news|nntp|sip|sips|tel|urn|webcal|xri|xmpp):'
    )
    
    # Check if URL starts with any dangerous scheme
    if re.match(dangerous_schemes, cleaned_url, re.IGNORECASE):
        return True
    
    # No dangerous schemes detected
    return False
