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


def is_internal_network(hostname: str) -> bool:
    """
    Check if the hostname matches any internal network pattern.
    
    Args:
        hostname: The hostname to check
        
    Returns:
        True if hostname matches an internal network pattern, False otherwise
    """
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False


def is_trusted_provider(hostname: str) -> bool:
    """
    Check if the hostname is in the list of trusted RSS providers.
    
    Args:
        hostname: The hostname to check
        
    Returns:
        True if hostname is a trusted provider, False otherwise
    """
    return hostname in TRUSTED_RSS_PROVIDERS


def check_feed_url(rss_link: str) -> str:
    """
    Validates and sanitizes an RSS feed URL by ensuring it comes from a trusted provider
    and doesn't attempt to access internal networks.
    
    Args:
        rss_link: The RSS feed URL that needs validation
        
    Returns:
        A valid, sanitized RSS feed URL
        
    Raises:
        ValueError: If the URL is not from a trusted provider or attempts connection to a restricted network area
    """
    # Initialize logger
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    logger.info(f"Starting URL validation for: {rss_link}")
    
    # Parse the URL
    try:
        parsed_url = urlparse(rss_link)
    except Exception as e:
        logger.error(f"Failed to parse URL '{rss_link}': {str(e)}")
        raise ValueError("Invalid URL format")
    
    # Check if the scheme is allowed
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        logger.error(f"Invalid URL scheme '{parsed_url.scheme}' for URL: {rss_link}")
        raise ValueError(f"URL scheme must be one of {ALLOWED_SCHEMES}")
    
    # Extract the hostname
    hostname = parsed_url.hostname
    if not hostname:
        logger.error(f"No hostname found in URL: {rss_link}")
        raise ValueError("URL must have a valid hostname")
    
    logger.debug(f"Extracted hostname: {hostname}")
    
    # Check against internal IP patterns
    if is_internal_network(hostname):
        logger.warning(f"Attempted access to internal network with hostname '{hostname}' from URL: {rss_link}")
        raise ValueError("Access to internal networks is not allowed")
    
    # Check if hostname is in the trusted providers list
    if not is_trusted_provider(hostname):
        logger.warning(f"Untrusted provider '{hostname}' attempted from URL: {rss_link}")
        raise ValueError(f"RSS feed must be from one of the trusted providers: {TRUSTED_RSS_PROVIDERS}")
    
    # Return the original URL if all checks pass
    logger.info(f"URL validation successful for: {rss_link}")
    return rss_link
