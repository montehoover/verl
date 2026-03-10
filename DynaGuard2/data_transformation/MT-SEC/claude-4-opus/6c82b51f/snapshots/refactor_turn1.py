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
    # Parse the URL
    try:
        parsed_url = urlparse(rss_link)
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
    
    # Check if hostname is in the trusted providers list
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"RSS feed must be from one of the trusted providers: {TRUSTED_RSS_PROVIDERS}")
    
    # Return the original URL if all checks pass
    return rss_link
