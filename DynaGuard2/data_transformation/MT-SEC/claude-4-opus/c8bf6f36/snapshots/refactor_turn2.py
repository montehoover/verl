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
    # Parse the URL
    try:
        parsed_url = urlparse(rss_link)
    except Exception:
        raise ValueError("Invalid URL format")
    
    # Validate URL scheme
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme must be one of {ALLOWED_SCHEMES}")
    
    # Extract and validate hostname
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("URL must have a valid hostname")
    
    # Check for internal network access
    if is_internal_network(hostname):
        raise ValueError("Access to internal networks is not allowed")
    
    # Verify trusted provider
    if not is_trusted_provider(hostname):
        raise ValueError(f"RSS provider '{hostname}' is not in the list of trusted providers")
    
    return rss_link
