import re
from urllib.parse import urlparse, ParseResult
from typing import Optional

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]


def parse_url(url: str) -> ParseResult:
    """Parse the URL and return the parsed result."""
    return urlparse(url)


def validate_scheme(parsed_url: ParseResult) -> None:
    """Validate that the URL scheme is allowed."""
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme. Only {', '.join(ALLOWED_SCHEMES)} are allowed.")


def extract_hostname(parsed_url: ParseResult) -> str:
    """Extract and validate the hostname from the parsed URL."""
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("Invalid URL: No hostname found.")
    return hostname


def validate_not_internal(hostname: str) -> None:
    """Validate that the hostname is not an internal network address."""
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError("Access to internal network addresses is not allowed.")


def validate_trusted_provider(hostname: str) -> None:
    """Validate that the hostname is from a trusted RSS provider."""
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL must be from a trusted RSS provider. Allowed providers: {', '.join(TRUSTED_RSS_PROVIDERS)}")


def clean_feed_link(rss_link: str) -> str:
    """
    Clean and validate an RSS feed URL.
    
    Args:
        rss_link: The RSS feed URL to validate
        
    Returns:
        The sanitized feed URL if valid and trusted
        
    Raises:
        ValueError: If the URL is not from a trusted source or attempts to access internal networks
    """
    # Pipeline of validation steps
    parsed_url = parse_url(rss_link)
    validate_scheme(parsed_url)
    hostname = extract_hostname(parsed_url)
    validate_not_internal(hostname)
    validate_trusted_provider(hostname)
    
    # Return the sanitized URL
    return rss_link
