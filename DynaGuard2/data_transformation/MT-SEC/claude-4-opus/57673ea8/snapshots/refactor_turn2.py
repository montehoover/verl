import re
from urllib.parse import urlparse

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
    return hostname in TRUSTED_RSS_PROVIDERS


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
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Check if the scheme is allowed
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme. Only {', '.join(ALLOWED_SCHEMES)} are allowed.")
    
    # Extract the hostname
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("Invalid URL: No hostname found.")
    
    # Check for internal IP patterns
    if is_internal_address(hostname):
        raise ValueError("Access to internal network addresses is not allowed.")
    
    # Check if the hostname is in the trusted providers list
    if not is_trusted_provider(hostname):
        raise ValueError(f"URL must be from a trusted RSS provider. Allowed providers: {', '.join(TRUSTED_RSS_PROVIDERS)}")
    
    return url
