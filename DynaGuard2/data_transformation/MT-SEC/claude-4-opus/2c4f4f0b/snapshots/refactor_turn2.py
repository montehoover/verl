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

def validate_url_scheme(parsed_url):
    """Validate that the URL uses an allowed scheme."""
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme. Only {', '.join(ALLOWED_SCHEMES)} are allowed.")
    return parsed_url

def validate_hostname_exists(parsed_url):
    """Validate that the URL has a hostname."""
    if not parsed_url.hostname:
        raise ValueError("Invalid URL: no hostname found.")
    return parsed_url

def validate_not_internal_address(parsed_url):
    """Validate that the hostname is not an internal network address."""
    hostname = parsed_url.hostname
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError("Access to internal network addresses is not allowed.")
    return parsed_url

def validate_trusted_provider(parsed_url):
    """Validate that the hostname is from a trusted RSS provider."""
    hostname = parsed_url.hostname
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL must be from a trusted RSS provider. Allowed providers: {', '.join(TRUSTED_RSS_PROVIDERS)}")
    return parsed_url

def check_rss_source(source_url: str) -> str:
    # Parse the URL
    parsed_url = urlparse(source_url)
    
    # Run validation pipeline
    validate_url_scheme(parsed_url)
    validate_hostname_exists(parsed_url)
    validate_not_internal_address(parsed_url)
    validate_trusted_provider(parsed_url)
    
    return source_url
