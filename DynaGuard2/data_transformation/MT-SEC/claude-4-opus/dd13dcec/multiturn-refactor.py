import re
import logging
from urllib.parse import urlparse

# Configure logger
logger = logging.getLogger(__name__)

TRUSTED_RSS_PROVIDERS = [
    "rss.trustedsource.com",
    "feeds.provider.com",
    "newsfeed.safesite.org"
]

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                       # Localhost
    r"^127\.0\.0\.\d+$",                 # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$", # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"     # Private 192.168.x.x range
]

ALLOWED_SCHEMES = ["http", "https"]


def is_valid_scheme(scheme: str) -> bool:
    """Check if the URL scheme is allowed."""
    return scheme in ALLOWED_SCHEMES


def is_internal_address(hostname: str) -> bool:
    """Check if the hostname matches internal IP patterns."""
    return any(re.match(pattern, hostname) for pattern in INTERNAL_IP_PATTERNS)


def is_trusted_provider(hostname: str) -> bool:
    """Check if the hostname belongs to trusted RSS providers."""
    return hostname in TRUSTED_RSS_PROVIDERS


def validate_url_scheme(parsed_url) -> None:
    """
    Validate that the URL scheme is allowed.
    
    Raises:
        ValueError: If the URL scheme is not allowed.
    """
    if not is_valid_scheme(parsed_url.scheme):
        allowed_schemes = ', '.join(ALLOWED_SCHEMES)
        error_msg = f"Invalid URL scheme '{parsed_url.scheme}'. Only {allowed_schemes} are allowed."
        logger.warning(f"URL scheme validation failed: {error_msg}")
        raise ValueError(error_msg)
    
    logger.debug(f"URL scheme '{parsed_url.scheme}' is valid")


def validate_hostname(hostname: str) -> None:
    """
    Validate that the hostname exists and is not an internal address.
    
    Args:
        hostname: The hostname to validate.
        
    Raises:
        ValueError: If hostname is None, internal, or not trusted.
    """
    if not hostname:
        error_msg = "Invalid URL: No hostname found."
        logger.error(f"Hostname validation failed: {error_msg}")
        raise ValueError(error_msg)
    
    logger.debug(f"Validating hostname: {hostname}")
    
    if is_internal_address(hostname):
        error_msg = f"Access to internal network address '{hostname}' is not allowed."
        logger.warning(f"Security violation attempt: {error_msg}")
        raise ValueError("Access to internal network addresses is not allowed.")
    
    if not is_trusted_provider(hostname):
        error_msg = f"URL host '{hostname}' is not in the list of trusted RSS providers."
        logger.warning(f"Untrusted provider: {error_msg}")
        raise ValueError(error_msg)
    
    logger.debug(f"Hostname '{hostname}' is a trusted provider")


def sanitize_feed_url(feed_url: str) -> str:
    """
    Sanitize and validate an RSS feed URL to ensure it only accesses trusted providers.
    
    Args:
        feed_url: str, the RSS feed URL to be sanitized
        
    Returns:
        A sanitized and validated RSS feed URL
        
    Raises:
        ValueError: If the URL does not belong to trusted providers or 
                   attempts to access restricted network URLs
    """
    logger.info(f"Starting validation for URL: {feed_url}")
    
    try:
        parsed_url = urlparse(feed_url)
        
        validate_url_scheme(parsed_url)
        validate_hostname(parsed_url.hostname)
        
        logger.info(f"URL validation successful: {feed_url}")
        return feed_url
        
    except ValueError as e:
        logger.error(f"URL validation failed for '{feed_url}': {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during URL validation for '{feed_url}': {str(e)}")
        raise ValueError(f"Invalid URL: {str(e)}")
