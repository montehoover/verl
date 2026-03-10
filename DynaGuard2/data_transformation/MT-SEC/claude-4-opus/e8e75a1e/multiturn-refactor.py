import re
import logging

# Configure logger
logger = logging.getLogger(__name__)


def _matches_dangerous_pattern(url_string: str, url_lower: str) -> bool:
    """
    Check if the URL matches any dangerous patterns.
    
    Args:
        url_string: The original URL string
        url_lower: The lowercase version of the URL string
        
    Returns:
        True if any dangerous pattern is found, False otherwise
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
    
    # Check for dangerous schemes
    for scheme in dangerous_schemes:
        if url_lower.startswith(scheme):
            logger.warning(f"Dangerous scheme detected: {scheme} in URL: {url_string}")
            return True
    
    try:
        # Check for obfuscated javascript: schemes
        if re.search(r'java[\s\r\n\t]*script\s*:', url_lower):
            logger.warning(f"Obfuscated javascript scheme detected in URL: {url_string}")
            return True
        
        # Check for vbscript with obfuscation
        if re.search(r'vb[\s\r\n\t]*script\s*:', url_lower):
            logger.warning(f"Obfuscated vbscript scheme detected in URL: {url_string}")
            return True
        
        # Check for encoded versions of javascript:
        if re.search(r'(%6[jJ]|j)(%61|a)(%76|v)(%61|a)(%73|s)(%63|c)(%72|r)(%69|i)(%70|p)(%74|t)(%3[aA]|:)', url_string):
            logger.warning(f"Encoded javascript scheme detected in URL: {url_string}")
            return True
    except re.error as e:
        logger.error(f"Regex error while checking URL: {url_string}, Error: {str(e)}")
        raise
    
    return False


def detect_malicious_url_scheme(url_string: str) -> bool:
    """
    Determine whether a given string contains a risky URL scheme.
    
    Args:
        url_string: The input string to be inspected for potentially malicious URL schemes
        
    Returns:
        True if the string contains any dangerous URL schemes (ignoring valid image data URLs);
        otherwise, False
    """
    logger.debug(f"Checking URL for malicious schemes: {url_string}")
    
    # Convert to lowercase for case-insensitive matching
    url_lower = url_string.lower().strip()
    
    try:
        # Guard clause: Check for valid image data URLs first
        if url_lower.startswith('data:') and re.match(r'^data:image/(jpeg|jpg|png|gif|bmp|webp|svg\+xml|x-icon|vnd\.microsoft\.icon)', url_lower):
            logger.info(f"Valid image data URL detected: {url_string}")
            return False
    except re.error as e:
        logger.error(f"Regex error while checking image data URL: {url_string}, Error: {str(e)}")
        raise
    
    # Check for dangerous patterns
    result = _matches_dangerous_pattern(url_string, url_lower)
    
    if result:
        logger.warning(f"Malicious URL scheme detected: {url_string}")
    else:
        logger.info(f"URL appears safe: {url_string}")
    
    return result
