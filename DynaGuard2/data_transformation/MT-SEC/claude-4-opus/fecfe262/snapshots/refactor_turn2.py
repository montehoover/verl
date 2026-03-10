import re
from urllib.parse import urlparse, ParseResult

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]


def parse_url(url: str) -> ParseResult:
    """Parse and validate URL format."""
    try:
        parsed = urlparse(url)
        if not parsed.hostname:
            raise ValueError("URL must have a valid hostname")
        return parsed
    except Exception:
        raise ValueError("Invalid URL format")


def validate_scheme(parsed_url: ParseResult) -> ParseResult:
    """Validate URL scheme is allowed."""
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme must be one of {ALLOWED_SCHEMES}")
    return parsed_url


def check_internal_network(parsed_url: ParseResult) -> ParseResult:
    """Check URL doesn't point to internal network."""
    hostname = parsed_url.hostname
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError("Access to internal networks is not allowed")
    return parsed_url


def validate_trusted_provider(parsed_url: ParseResult) -> ParseResult:
    """Validate hostname is from trusted provider."""
    if parsed_url.hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"RSS feed must be from one of the trusted providers: {TRUSTED_RSS_PROVIDERS}")
    return parsed_url


def filter_rss_feed(rss_feed_url: str) -> str:
    """
    Validates and sanitizes an RSS feed URL to ensure it points only to trusted providers.
    
    Args:
        rss_feed_url: The URL of the RSS feed being validated
        
    Returns:
        A properly sanitized feed URL limited to trusted providers
        
    Raises:
        ValueError: If the input URL is not from an approved provider or if it attempts 
                   access to restricted internal networks
    """
    # Pipeline of validation functions
    pipeline = [
        parse_url,
        validate_scheme,
        check_internal_network,
        validate_trusted_provider
    ]
    
    # Run URL through validation pipeline
    result = rss_feed_url
    for step in pipeline:
        if step == parse_url:
            result = step(result)
        else:
            result = step(result)
    
    # Return the original URL if all checks pass
    return rss_feed_url
