import re
import logging
from urllib.parse import urlparse

# Initialize logger
logger = logging.getLogger(__name__)
# Basic configuration for demonstration.
# In a real application, logging would be configured externally.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]


def _parse_url(rss_link: str) -> urlparse.ParseResult:
    """Parses the URL string."""
    try:
        parsed_url = urlparse(rss_link)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {rss_link}. Error: {e}")
    return parsed_url


def _validate_scheme(parsed_url: urlparse.ParseResult) -> urlparse.ParseResult:
    """Validates the URL scheme."""
    if not parsed_url.scheme or parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}")
    return parsed_url


def _validate_hostname(parsed_url: urlparse.ParseResult) -> urlparse.ParseResult:
    """Validates that the URL has a hostname."""
    if not parsed_url.hostname:
        # Pass the original rss_link for the error message if available,
        # otherwise construct from parsed_url. This assumes clean_feed_link context.
        # For a truly pure function, it might only have access to parsed_url.
        # However, the original error message used rss_link.
        original_url_for_error = parsed_url.geturl() if parsed_url.geturl() else "the provided URL"
        raise ValueError(f"URL must contain a valid hostname: {original_url_for_error}")
    return parsed_url


def _check_internal_ip(parsed_url: urlparse.ParseResult) -> urlparse.ParseResult:
    """Checks if the hostname matches any internal IP patterns."""
    hostname = parsed_url.hostname # Already validated to exist by _validate_hostname
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal network address is forbidden: {hostname}")
    return parsed_url


def _check_trusted_provider(parsed_url: urlparse.ParseResult) -> urlparse.ParseResult:
    """Checks if the hostname is in the list of trusted providers."""
    hostname = parsed_url.hostname # Already validated to exist
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL hostname '{hostname}' is not in the list of trusted providers.")
    return parsed_url


def clean_feed_link(rss_link: str) -> str:
    """
    Ensures an RSS feed URL comes from predefined trusted providers and does not point to an internal network.

    Args:
        rss_link: The RSS feed URL that requires sanitization.

    Returns:
        The sanitized feed URL if valid and trusted.

    Raises:
        ValueError: If the provided URL does not come from an authenticated source,
                    uses a disallowed scheme, or if access to an internal network address is detected.
    """
    logger.info(f"Attempting to clean and validate URL: {rss_link}")
    try:
        parsed_url = _parse_url(rss_link)
        _validate_scheme(parsed_url)
        _validate_hostname(parsed_url)  # Ensures hostname exists for subsequent checks
        _check_internal_ip(parsed_url)
        _check_trusted_provider(parsed_url)
        
        logger.info(f"URL validation successful for: {rss_link}")
        return rss_link
    except ValueError as e:
        logger.error(f"URL validation failed for {rss_link}: {e}")
        raise
    except Exception as e: # Catch any other unexpected errors during parsing/validation
        logger.error(f"An unexpected error occurred during validation of {rss_link}: {e}")
        raise ValueError(f"An unexpected error occurred during validation of {rss_link}.")
