import re
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]


def _parse_and_validate_basic_url(rss_feed_url: str) -> urlparse.ParseResult:
    """Parses the URL and ensures a hostname is present."""
    logger.info(f"Attempting to parse URL: {rss_feed_url}")
    try:
        parsed_url = urlparse(rss_feed_url)
    except Exception as e:
        logger.error(f"Failed to parse URL: {rss_feed_url}. Error: {e}")
        raise ValueError(f"Invalid URL format: {rss_feed_url}") from e

    if not parsed_url.hostname:
        logger.error(f"URL validation failed: No hostname present in {rss_feed_url}")
        raise ValueError("URL must have a hostname.")
    logger.info(f"Successfully parsed URL: {rss_feed_url}. Hostname: {parsed_url.hostname}")
    return parsed_url


def _validate_url_scheme(parsed_url: urlparse.ParseResult) -> urlparse.ParseResult:
    """Validates the URL's scheme."""
    logger.info(f"Validating URL scheme for: {parsed_url.geturl()}")
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        logger.error(f"Scheme validation failed for {parsed_url.geturl()}. Scheme: {parsed_url.scheme}. Allowed: {ALLOWED_SCHEMES}")
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Must be one of {ALLOWED_SCHEMES}.")
    logger.info(f"Scheme validation successful for {parsed_url.geturl()}. Scheme: {parsed_url.scheme}")
    return parsed_url


def _validate_hostname_not_internal(parsed_url: urlparse.ParseResult) -> urlparse.ParseResult:
    """Ensures the hostname is not an internal IP."""
    hostname = parsed_url.hostname
    logger.info(f"Validating hostname against internal IP patterns for: {hostname}")
    # Hostname presence is already checked by _parse_and_validate_basic_url
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            logger.error(f"Internal IP check failed for hostname: {hostname}. Matched pattern: {pattern}")
            raise ValueError(f"Access to internal network resource is forbidden: {hostname}")
    logger.info(f"Hostname {hostname} is not an internal IP.")
    return parsed_url


def _validate_hostname_is_trusted(parsed_url: urlparse.ParseResult) -> urlparse.ParseResult:
    """Ensures the hostname is a trusted provider."""
    hostname = parsed_url.hostname
    logger.info(f"Validating if hostname {hostname} is a trusted provider.")
    # Hostname presence is already checked by _parse_and_validate_basic_url
    if hostname not in TRUSTED_RSS_PROVIDERS:
        logger.error(f"Trusted provider check failed for hostname: {hostname}. Not in {TRUSTED_RSS_PROVIDERS}")
        raise ValueError(f"URL provider not trusted: {hostname}. Must be one of {TRUSTED_RSS_PROVIDERS}.")
    logger.info(f"Hostname {hostname} is a trusted provider.")
    return parsed_url


def filter_rss_feed(rss_feed_url: str) -> str:
    """
    Validates an RSS feed URL to ensure it's from a trusted provider
    and does not point to internal networks.

    Args:
        rss_feed_url: The URL of the RSS feed.

    Returns:
        The sanitized feed URL if valid.

    Raises:
        ValueError: If the URL is invalid, not from an approved provider,
                    or attempts to access restricted internal networks.
    """
    logger.info(f"Starting RSS feed URL validation for: {rss_feed_url}")
    try:
        parsed_url = _parse_and_validate_basic_url(rss_feed_url)
        parsed_url = _validate_url_scheme(parsed_url)
        parsed_url = _validate_hostname_not_internal(parsed_url)
        _validate_hostname_is_trusted(parsed_url)  # Final check, result not needed beyond validation
        logger.info(f"Successfully validated RSS feed URL: {rss_feed_url}")
        return rss_feed_url
    except ValueError as e:
        logger.error(f"Validation failed for RSS feed URL: {rss_feed_url}. Error: {e}")
        raise
