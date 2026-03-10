import re
import logging

# Configure logger
logger = logging.getLogger(__name__)


def _is_image_data_url(url: str) -> bool:
    """
    Checks if the URL is a valid image data URL.
    
    Args:
        url: str, the URL to check
        
    Returns:
        bool, True if it's a valid image data URL, False otherwise
    """
    image_data_pattern = r'^data:image/(gif|jpeg|jpg|png|svg\+xml|webp);'
    return bool(re.match(image_data_pattern, url, re.IGNORECASE))


def _has_dangerous_scheme(url: str) -> bool:
    """
    Checks if the URL contains a dangerous scheme.
    
    Args:
        url: str, the URL to check
        
    Returns:
        bool, True if it contains a dangerous scheme, False otherwise
    """
    dangerous_schemes = (
        r'^(javascript|vbscript|data:(?!image/)[^;]+|about|chrome|'
        r'chrome-extension|ms-help|hcp|mk|mhtml|help|disk|afp|x-man-page|'
        r'x-help|ms-its|ms-itss|ldap|mailto|news|nntp|telnet|gopher|wais|file):'
    )
    return bool(re.match(dangerous_schemes, url, re.IGNORECASE))


def analyze_scheme(url: str) -> bool:
    """
    Analyzes a given string to determine if it contains a potentially harmful URL scheme.
    
    Args:
        url: str, the string to examine for possible dangerous URL schemes.
        
    Returns:
        bool, returns True if the string involves a potentially harmful URL scheme,
        aside from valid image data URLs; returns False otherwise.
    """
    # Check if it's a valid image data URL first
    if _is_image_data_url(url):
        logger.info(f"URL processed: {url!r} - Result: Safe (valid image data URL)")
        return False
    
    # Check for dangerous schemes
    is_dangerous = _has_dangerous_scheme(url)
    
    if is_dangerous:
        logger.warning(f"URL processed: {url!r} - Result: Dangerous")
    else:
        logger.info(f"URL processed: {url!r} - Result: Safe")
    
    return is_dangerous
