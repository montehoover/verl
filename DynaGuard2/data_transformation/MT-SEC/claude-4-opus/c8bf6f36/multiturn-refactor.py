import re
import logging
from urllib.parse import urlparse

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]

# Configure logger
logger = logging.getLogger(__name__)


def is_internal_network(hostname: str) -> bool:
    """
    Check if a hostname matches internal network patterns.
    
    Args:
        hostname: The hostname to check
        
    Returns:
        True if the hostname matches an internal network pattern, False otherwise
    """
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False


def is_trusted_provider(hostname: str) -> bool:
    """
    Check if a hostname is in the list of trusted RSS providers.
    
    Args:
        hostname: The hostname to check
        
    Returns:
        True if the hostname is a trusted provider, False otherwise
    """
    return hostname in TRUSTED_RSS_PROVIDERS


def validate_rss_link(rss_link: str) -> str:
    """
    Validate that an RSS feed URL points only to trusted providers.
    
    Args:
        rss_link: The URL of the RSS feed being validated
        
    Returns:
        A properly sanitized feed URL limited to trusted providers
        
    Raises:
        ValueError: If the input URL is not from an approved provider or 
                   if it attempts access to restricted internal networks
    """
    logger.info(f"Starting validation for RSS link: {rss_link}")
    
    # Parse the URL
    try:
        parsed_url = urlparse(rss_link)
    except Exception as e:
        logger.error(f"Failed to parse URL '{rss_link}': {str(e)}")
        raise ValueError("Invalid URL format")
    
    # Validate URL scheme
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        error_msg = f"URL scheme must be one of {ALLOWED_SCHEMES}"
        logger.warning(f"Validation failed for '{rss_link}': {error_msg}")
        raise ValueError(error_msg)
    
    # Extract and validate hostname
    hostname = parsed_url.hostname
    if not hostname:
        logger.warning(f"Validation failed for '{rss_link}': No valid hostname found")
        raise ValueError("URL must have a valid hostname")
    
    # Check for internal network access
    if is_internal_network(hostname):
        logger.warning(f"Validation failed for '{rss_link}': Attempted access to internal network '{hostname}'")
        raise ValueError("Access to internal networks is not allowed")
    
    # Verify trusted provider
    if not is_trusted_provider(hostname):
        error_msg = f"RSS provider '{hostname}' is not in the list of trusted providers"
        logger.warning(f"Validation failed for '{rss_link}': {error_msg}")
        raise ValueError(error_msg)
    
    logger.info(f"Validation successful for RSS link: {rss_link}")
    return rss_link
