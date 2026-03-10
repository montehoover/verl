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
    # Parse the URL
    parsed_url = urlparse(source_url)
    
    # Check if the scheme is allowed
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme. Only {', '.join(ALLOWED_SCHEMES)} are allowed.")
    
    # Get the hostname
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("Invalid URL: no hostname found.")
    
    # Check if the hostname matches internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError("Access to internal network addresses is not allowed.")
    
    # Check if the hostname is from a trusted provider
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL must be from a trusted RSS provider. Allowed providers: {', '.join(TRUSTED_RSS_PROVIDERS)}")
    
    return source_url
