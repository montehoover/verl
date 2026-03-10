import re

def verify_javascript_protocol(url_input: str) -> bool:
    """
    Analyzes a given string to determine if it contains a potentially harmful URL scheme.
    
    Args:
        url_input: str, the string to examine for possible dangerous URL schemes
        
    Returns:
        bool: returns True if the string involves a potentially harmful URL scheme,
        aside from valid image data URLs; returns False otherwise
    """
    # Pattern to match dangerous URL schemes
    dangerous_schemes = r'^(javascript|vbscript|data:(?!image/)[^;]+|about|chrome-extension|moz-extension|ms-browser-extension|opera|res|resource|moz-icon|android-app|app|blob|content|intent|moz-safe-about|moz-extension|sftp|ssh|ftp|ftps|webdav|webdavs|dav|davs|afp|nfs|smb|dav-noredirect|jar|xmpp|xmpp-bosh|sms|tel|wtai|market|gopher|mms|nfs|nntp|telnet|imap|rtsp|tag|sftp|ssh|bitcoin|magnet|aim|apt|attachment|cid|data|dav|dns|fax|file|finger|gopher|im|imap|ldap|mailto|news|nntp|sip|sips|tel|urn|webcal|xri|xmpp):'
    
    # Check if the input matches any dangerous scheme
    if re.match(dangerous_schemes, url_input.strip(), re.IGNORECASE):
        # Special case: allow image data URLs
        if re.match(r'^data:image/', url_input.strip(), re.IGNORECASE):
            return False
        return True
    
    return False
