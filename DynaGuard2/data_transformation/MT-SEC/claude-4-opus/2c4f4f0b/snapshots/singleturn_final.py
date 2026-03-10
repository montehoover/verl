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

def check_rss_source(source_url: str) -> str:
    """
    Validate RSS feed URL to ensure it comes from trusted providers.
    
    Args:
        source_url: The RSS feed URL that requires checking
        
    Returns:
        The checked feed URL if valid and trusted
        
    Raises:
        ValueError: If the URL is not from a trusted source or targets internal network
    """
    # Parse the URL
    try:
        parsed_url = urlparse(source_url)
    except Exception:
        raise ValueError("Invalid URL format")
    
    # Check if scheme is allowed
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme}. Only {ALLOWED_SCHEMES} are allowed")
    
    # Extract hostname
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("No hostname found in URL")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal network address '{hostname}' is not allowed")
    
    # Check if hostname is in trusted providers
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"RSS source '{hostname}' is not in the list of trusted providers")
    
    return source_url
