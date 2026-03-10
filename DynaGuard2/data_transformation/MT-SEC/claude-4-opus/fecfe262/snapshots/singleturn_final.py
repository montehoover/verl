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


def filter_rss_feed(rss_feed_url: str) -> str:
    """
    Validate and sanitize an RSS feed URL to ensure it comes from trusted providers.
    
    Args:
        rss_feed_url: The URL of the RSS feed being validated
        
    Returns:
        A properly sanitized feed URL limited to trusted providers
        
    Raises:
        ValueError: If the input URL is not from an approved provider or if it 
                   attempts access to restricted internal networks
    """
    # Parse the URL
    try:
        parsed_url = urlparse(rss_feed_url)
    except Exception:
        raise ValueError("Invalid URL format")
    
    # Check if the scheme is allowed
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme must be one of {ALLOWED_SCHEMES}")
    
    # Extract the hostname
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("URL must have a valid hostname")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError("Access to internal networks is not allowed")
    
    # Check if the hostname is in the trusted providers list
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"RSS provider '{hostname}' is not in the list of trusted providers")
    
    # Return the original URL if all validations pass
    return rss_feed_url
