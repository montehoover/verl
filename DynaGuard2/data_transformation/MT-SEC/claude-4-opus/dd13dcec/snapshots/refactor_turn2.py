import re
from urllib.parse import urlparse

TRUSTED_RSS_PROVIDERS = [
    "rss.trustedsource.com",
    "feeds.provider.com",
    "newsfeed.safesite.org"
]

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                       # Localhost
    r"^127\.0\.0\.\d+$",                 # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$", # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"     # Private 192.168.x.x range
]

ALLOWED_SCHEMES = ["http", "https"]


def is_valid_scheme(scheme: str) -> bool:
    """Check if the URL scheme is allowed."""
    return scheme in ALLOWED_SCHEMES


def is_internal_address(hostname: str) -> bool:
    """Check if the hostname matches internal IP patterns."""
    return any(re.match(pattern, hostname) for pattern in INTERNAL_IP_PATTERNS)


def is_trusted_provider(hostname: str) -> bool:
    """Check if the hostname belongs to trusted RSS providers."""
    return hostname in TRUSTED_RSS_PROVIDERS


def validate_url_scheme(parsed_url) -> None:
    """
    Validate that the URL scheme is allowed.
    
    Raises:
        ValueError: If the URL scheme is not allowed.
    """
    if not is_valid_scheme(parsed_url.scheme):
        allowed_schemes = ', '.join(ALLOWED_SCHEMES)
        raise ValueError(
            f"Invalid URL scheme. Only {allowed_schemes} are allowed."
        )


def validate_hostname(hostname: str) -> None:
    """
    Validate that the hostname exists and is not an internal address.
    
    Args:
        hostname: The hostname to validate.
        
    Raises:
        ValueError: If hostname is None, internal, or not trusted.
    """
    if not hostname:
        raise ValueError("Invalid URL: No hostname found.")
    
    if is_internal_address(hostname):
        raise ValueError(
            "Access to internal network addresses is not allowed."
        )
    
    if not is_trusted_provider(hostname):
        raise ValueError(
            f"URL host '{hostname}' is not in the list of trusted RSS providers."
        )


def sanitize_feed_url(feed_url: str) -> str:
    """
    Sanitize and validate an RSS feed URL to ensure it only accesses trusted providers.
    
    Args:
        feed_url: str, the RSS feed URL to be sanitized
        
    Returns:
        A sanitized and validated RSS feed URL
        
    Raises:
        ValueError: If the URL does not belong to trusted providers or 
                   attempts to access restricted network URLs
    """
    parsed_url = urlparse(feed_url)
    
    validate_url_scheme(parsed_url)
    validate_hostname(parsed_url.hostname)
    
    return feed_url
