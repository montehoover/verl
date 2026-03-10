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


def clean_feed_link(rss_link: str) -> str:
    """
    Validates and sanitizes an RSS feed URL to ensure it comes from trusted providers.
    
    Args:
        rss_link: str - The RSS feed URL that requires sanitization
        
    Returns:
        str - The sanitized feed URL if valid and trusted
        
    Raises:
        ValueError - If the provided URL does not come from an authenticated source
                    or if access to an internal network address is detected
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
            raise ValueError("Access to internal network addresses is not allowed")
    
    # Check if it's an IP address (any IP that's not already caught by internal patterns)
    # This helps prevent access to other private IPs not covered by our patterns
    ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    if re.match(ip_pattern, hostname):
        raise ValueError("Direct IP addresses are not allowed, only trusted domain names")
    
    # Check if the hostname is in the trusted providers list
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"RSS provider '{hostname}' is not in the list of trusted providers")
    
    # Return the original URL if all checks pass
    return rss_link
