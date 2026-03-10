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

def sanitize_feed_url(feed_url: str) -> str:
    """
    Sanitize and validate an RSS feed URL to ensure it only accesses trusted providers.
    
    Args:
        feed_url: str, the RSS feed URL to be sanitized
        
    Returns:
        A sanitized and validated RSS feed URL
        
    Raises:
        ValueError: If the URL does not belong to trusted providers or attempts to access restricted network URLs
    """
    # Parse the URL
    parsed_url = urlparse(feed_url)
    
    # Check if the scheme is allowed
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme. Only {', '.join(ALLOWED_SCHEMES)} are allowed.")
    
    # Extract hostname
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("Invalid URL: No hostname found.")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError("Access to internal network addresses is not allowed.")
    
    # Check if hostname is in trusted providers
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL host '{hostname}' is not in the list of trusted RSS providers.")
    
    # Return the original URL if all checks pass
    return feed_url
