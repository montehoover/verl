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

def validate_feed_url(url: str) -> str:
    """
    Validates an RSS feed URL to ensure it comes from trusted providers.
    
    Args:
        url: str - The RSS feed URL that requires validation
        
    Returns:
        str - The validated feed URL if valid and trusted
        
    Raises:
        ValueError - If the URL is not from a trusted source or points to internal network
    """
    # Parse the URL
    try:
        parsed = urlparse(url)
    except Exception:
        raise ValueError("Invalid URL format")
    
    # Check if the scheme is allowed
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme must be one of {ALLOWED_SCHEMES}")
    
    # Extract hostname
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must have a valid hostname")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname, re.IGNORECASE):
            raise ValueError("Access to internal network addresses is not allowed")
    
    # Check if hostname is in trusted providers
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL must be from one of the trusted providers: {TRUSTED_RSS_PROVIDERS}")
    
    return url
