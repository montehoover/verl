import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def has_script_scheme(url: str) -> bool:
    """
    Determine whether a given string contains a risky URL scheme.
    
    Args:
        url: The input string to be inspected for potentially malicious URL schemes.
        
    Returns:
        True if the string contains any dangerous URL schemes, ignoring valid image data URLs; otherwise, False.
    """
    logger.info(f"Checking URL for risky schemes: {url[:100]}{'...' if len(url) > 100 else ''}")
    
    normalized_url = _normalize_url(url)
    
    if _has_dangerous_scheme(normalized_url):
        logger.warning(f"Dangerous scheme detected in URL: {normalized_url[:100]}{'...' if len(normalized_url) > 100 else ''}")
        return True
    
    if _has_encoded_javascript(normalized_url):
        logger.warning(f"Encoded JavaScript detected in URL: {normalized_url[:100]}{'...' if len(normalized_url) > 100 else ''}")
        return True
        
    if _has_encoded_dangerous_schemes(normalized_url):
        logger.warning(f"Encoded dangerous scheme detected in URL: {normalized_url[:100]}{'...' if len(normalized_url) > 100 else ''}")
        return True
    
    logger.info(f"URL is safe: {normalized_url[:100]}{'...' if len(normalized_url) > 100 else ''}")
    return False


def _normalize_url(url: str) -> str:
    """Normalize the URL by stripping whitespace."""
    normalized = url.strip()
    logger.debug(f"Normalized URL from '{url}' to '{normalized}'")
    return normalized


def _has_dangerous_scheme(url: str) -> bool:
    """Check if URL starts with a dangerous scheme."""
    dangerous_schemes_pattern = _get_dangerous_schemes_pattern()
    match = re.match(dangerous_schemes_pattern, url, re.IGNORECASE)
    if match:
        logger.debug(f"Matched dangerous scheme: {match.group()}")
    return bool(match)


def _get_dangerous_schemes_pattern() -> str:
    """Return regex pattern for dangerous URL schemes."""
    schemes = [
        'javascript', 'vbscript', 'data:(?!image/)[^;]+', 'about', 'chrome',
        'chrome-extension', 'filesystem', 'blob', 'resource', 'ftp', 'sftp',
        'mailto', 'news', 'telnet', 'ldap', 'ldaps', 'ssh', 'irc', 'nntp',
        'worldwind', 'webcal', 'ms-help', 'help', 'disk', 'disks', 'hcp',
        'iehistory', 'ierss', 'iesetup', 'iexplore', 'jstemplate', 'mk',
        'mhtml', 'mht', 'opera', 'res', 'rlogin', 'shell', 'vnd\.ms\.radio',
        'vview', 'ms-itss', 'mso', 'its', 'msdun', 'ieframe', 'ms-its',
        'ms-shell', 'search', 'search-ms', 'sysimage', 'view-source',
        'vbscript\s*:'
    ]
    return r'^(' + '|'.join(schemes) + ')'


def _has_encoded_javascript(url: str) -> bool:
    """Check for encoded javascript schemes."""
    match = re.search(r'javascript\s*:', url, re.IGNORECASE)
    if match:
        logger.debug(f"Found encoded JavaScript pattern: {match.group()}")
    return bool(match)


def _has_encoded_dangerous_schemes(url: str) -> bool:
    """Check for URL-encoded or HTML-encoded dangerous schemes."""
    encoded_patterns = _get_encoded_patterns()
    
    for pattern in encoded_patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            logger.debug(f"Found encoded dangerous pattern: {pattern} matched {match.group()}")
            return True
    
    return False


def _get_encoded_patterns() -> list[str]:
    """Return list of encoded dangerous scheme patterns."""
    return [
        r'%6A%61%76%61%73%63%72%69%70%74',  # javascript
        r'%76%62%73%63%72%69%70%74',  # vbscript
        r'&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;',  # javascript HTML entities
        r'&#x6A;&#x61;&#x76;&#x61;&#x73;&#x63;&#x72;&#x69;&#x70;&#x74;',  # javascript hex entities
    ]
