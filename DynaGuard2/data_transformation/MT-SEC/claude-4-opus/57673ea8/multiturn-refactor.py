import re
import logging
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger(__name__)

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]


def is_internal_address(hostname: str) -> bool:
    """
    Check if the hostname matches any internal IP address patterns.
    
    Args:
        hostname: The hostname to check against internal IP patterns.
        
    Returns:
        True if the hostname matches an internal IP pattern, False otherwise.
    """
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            logger.debug(f"Hostname '{hostname}' matched internal IP pattern: {pattern}")
            return True
    return False


def is_trusted_provider(hostname: str) -> bool:
    """
    Check if the hostname is in the list of trusted RSS providers.
    
    Args:
        hostname: The hostname to check against trusted providers.
        
    Returns:
        True if the hostname is a trusted provider, False otherwise.
    """
    is_trusted = hostname in TRUSTED_RSS_PROVIDERS
    if is_trusted:
        logger.debug(f"Hostname '{hostname}' is a trusted provider")
    else:
        logger.debug(f"Hostname '{hostname}' is NOT a trusted provider")
    return is_trusted


def validate_feed_url(url: str) -> str:
    """
    Validates an RSS feed URL to ensure it comes from trusted providers.
    
    Args:
        url: The RSS feed URL that requires validation.
        
    Returns:
        The validated feed URL if valid and trusted.
        
    Raises:
        ValueError: If the URL doesn't come from an authenticated source or 
                   if access to an internal network address is detected.
    """
    logger.info(f"Starting validation for URL: {url}")
    
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Check if the scheme is allowed
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        error_msg = f"Invalid URL scheme '{parsed_url.scheme}'. Only {', '.join(ALLOWED_SCHEMES)} are allowed."
        logger.error(f"Validation failed for URL '{url}': {error_msg}")
        raise ValueError(error_msg)
    
    logger.debug(f"URL scheme '{parsed_url.scheme}' is valid")
    
    # Extract the hostname
    hostname = parsed_url.hostname
    if not hostname:
        error_msg = "Invalid URL: No hostname found."
        logger.error(f"Validation failed for URL '{url}': {error_msg}")
        raise ValueError(error_msg)
    
    logger.debug(f"Extracted hostname: {hostname}")
    
    # Check for internal IP patterns
    if is_internal_address(hostname):
        error_msg = "Access to internal network addresses is not allowed."
        logger.warning(f"Validation failed for URL '{url}': Attempted access to internal network address '{hostname}'")
        raise ValueError(error_msg)
    
    # Check if the hostname is in the trusted providers list
    if not is_trusted_provider(hostname):
        error_msg = f"URL must be from a trusted RSS provider. Allowed providers: {', '.join(TRUSTED_RSS_PROVIDERS)}"
        logger.warning(f"Validation failed for URL '{url}': Untrusted provider '{hostname}'")
        raise ValueError(error_msg)
    
    logger.info(f"Validation successful for URL: {url}")
    return url
