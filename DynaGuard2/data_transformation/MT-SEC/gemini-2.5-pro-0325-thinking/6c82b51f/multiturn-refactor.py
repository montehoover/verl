import logging
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

logger = logging.getLogger(__name__)


def _is_trusted_provider(hostname: str) -> bool:
    """Checks if the hostname is in the list of trusted providers."""
    return hostname.lower() in TRUSTED_RSS_PROVIDERS


def _is_internal_ip(hostname: str) -> bool:
    """Checks if the hostname matches any internal IP patterns."""
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False


def check_feed_url(rss_link: str) -> str:
    """
    Validates an RSS feed URL, ensuring it's from a trusted provider
    and does not access internal networks.

    Args:
        rss_link: The RSS feed URL to validate.

    Returns:
        The valid, sanitized RSS feed URL.

    Raises:
        ValueError: If the URL is invalid, not from a trusted provider,
                    or attempts to connect to a restricted network area.
    """
    logger.info(f"Attempting to validate URL: {rss_link}")
    try:
        parsed_url = urlparse(rss_link)
    except Exception as e:
        error_message = f"Invalid URL: {rss_link}. Error: {e}"
        logger.error(f"URL validation failed for {rss_link}: {error_message}")
        raise ValueError(error_message)

    if not parsed_url.scheme or not parsed_url.netloc:
        error_message = f"Invalid URL structure: {rss_link}"
        logger.error(f"URL validation failed for {rss_link}: {error_message}")
        raise ValueError(error_message)

    if parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        error_message = (
            f"Invalid URL scheme: {parsed_url.scheme}. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}"
        )
        logger.error(f"URL validation failed for {rss_link}: {error_message}")
        raise ValueError(error_message)

    hostname = parsed_url.hostname
    if hostname is None:
        error_message = f"Could not determine hostname from URL: {rss_link}"
        logger.error(f"URL validation failed for {rss_link}: {error_message}")
        raise ValueError(error_message)

    if not _is_trusted_provider(hostname):
        error_message = (
            f"URL provider not trusted: {hostname}. "
            f"Must be one of: {', '.join(TRUSTED_RSS_PROVIDERS)}"
        )
        logger.error(f"URL validation failed for {rss_link}: {error_message}")
        raise ValueError(error_message)

    if _is_internal_ip(hostname):
        error_message = (
            f"Access to internal or restricted network address is forbidden: {hostname}"
        )
        logger.error(f"URL validation failed for {rss_link}: {error_message}")
        raise ValueError(error_message)

    logger.info(f"URL validation successful for: {rss_link}")
    return rss_link
