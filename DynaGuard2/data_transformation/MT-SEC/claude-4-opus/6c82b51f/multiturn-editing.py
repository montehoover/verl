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

def check_feed_url(rss_link):
    """
    Validates an RSS feed URL to ensure it comes from a trusted provider
    and does not access internal networks.
    
    Args:
        rss_link (str): The RSS feed URL to validate
        
    Returns:
        str: The sanitized feed URL if valid
        
    Raises:
        ValueError: If the URL does not meet security criteria
    """
    # Parse the URL
    parsed = urlparse(rss_link)
    
    # Check if scheme is allowed
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid scheme: {parsed.scheme}. Only {', '.join(ALLOWED_SCHEMES)} are allowed.")
    
    # Get the hostname
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL: No hostname found.")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal network address '{hostname}' is not allowed.")
    
    # Check if hostname is from trusted provider
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"Untrusted RSS provider: {hostname}. Only trusted providers are allowed.")
    
    # Return the sanitized URL (reconstructed to ensure proper formatting)
    return rss_link
