import re

def detect_malicious_url_scheme(url_string: str) -> bool:
    """
    Determine whether a given string contains a risky URL scheme.
    
    Args:
        url_string: The input string to be inspected for potentially malicious URL schemes
        
    Returns:
        True if the string contains any dangerous URL schemes (ignoring valid image data URLs);
        otherwise, False
    """
    # Define dangerous URL schemes
    dangerous_schemes = [
        'javascript:', 'vbscript:', 'data:text/html', 'data:application',
        'about:', 'chrome:', 'chrome-extension:', 'ms-', 'shell:', 'webkit-',
        'opera:', 'firefox:', 'ie:', 'mhtml:', 'file:', 'res:', 'x-res:',
        'jar:', 'jnlp:', 'hcp:', 'ms-help:', 'help:', 'disk:', 'afp:',
        'disks:', 'telnet:', 'ssh:', 'ftp:', 'sftp:', 'ldap:', 'ldaps:',
        'mailto:', 'news:', 'nntp:', 'tel:', 'sms:', 'smsto:', 'snews:',
        'feed:', 'gopher:', 'urn:', 'view-source:', 'ws:', 'wss:'
    ]
    
    # Convert to lowercase for case-insensitive matching
    url_lower = url_string.lower().strip()
    
    # Check for dangerous schemes
    for scheme in dangerous_schemes:
        if url_lower.startswith(scheme):
            return True
    
    # Check for data: URLs that are NOT image types
    if url_lower.startswith('data:'):
        # Valid image data URLs start with data:image/
        if not re.match(r'^data:image/(jpeg|jpg|png|gif|bmp|webp|svg\+xml|x-icon|vnd\.microsoft\.icon)', url_lower):
            return True
    
    # Check for obfuscated javascript: schemes
    # Look for patterns like java\nscript:, java\rscript:, java\tscript:, etc.
    if re.search(r'java[\s\r\n\t]*script\s*:', url_lower):
        return True
    
    # Check for vbscript with obfuscation
    if re.search(r'vb[\s\r\n\t]*script\s*:', url_lower):
        return True
    
    # Check for encoded versions of javascript:
    # %6A%61%76%61%73%63%72%69%70%74%3A is "javascript:"
    if re.search(r'(%6[jJ]|j)(%61|a)(%76|v)(%61|a)(%73|s)(%63|c)(%72|r)(%69|i)(%70|p)(%74|t)(%3[aA]|:)', url_string):
        return True
    
    return False
