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


def _is_internal_ip(hostname: str) -> bool:
    """Check if the hostname matches any internal IP patterns."""
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False


def _is_trusted_provider(hostname: str) -> bool:
    """Check if the hostname is in the list of trusted providers."""
    return hostname in TRUSTED_RSS_PROVIDERS


def validate_rss_link(rss_link: str) -> str:
    """
    Ensure that the URL points only to trusted providers, thereby preventing
    access to insecure or internal networks.

    Args:
        rss_link: str, which is the URL of the RSS feed being validated.

    Returns:
        A properly sanitized feed URL limited to trusted providers.

    Raises:
        ValueError: If the input URL is not from an approved provider,
                    if it attempts access to restricted internal networks,
                    or if the URL scheme is not allowed.
    """
    logger.info(f"Attempting to validate RSS link: {rss_link}")
    try:
        parsed_url = urlparse(rss_link)
    except Exception as e:
        error_message = f"Invalid URL format: {rss_link}. Error: {e}"
        logger.error(error_message)
        raise ValueError(error_message)

    if not parsed_url.scheme or parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        error_message = (
            f"Invalid URL scheme: '{parsed_url.scheme}'. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}. "
            f"URL: {rss_link}"
        )
        logger.error(error_message)
        raise ValueError(error_message)

    hostname = parsed_url.hostname
    if not hostname:
        error_message = f"URL must have a hostname: {rss_link}"
        logger.error(error_message)
        raise ValueError(error_message)

    if _is_internal_ip(hostname):
        error_message = (
            f"Access to internal or restricted network denied for URL: {rss_link}"
        )
        logger.error(error_message)
        raise ValueError(error_message)

    if not _is_trusted_provider(hostname):
        error_message = (
            f"URL provider '{hostname}' is not in the list of trusted providers. "
            f"Trusted providers are: {', '.join(TRUSTED_RSS_PROVIDERS)}. "
            f"URL: {rss_link}"
        )
        logger.error(error_message)
        raise ValueError(error_message)

    logger.info(f"Successfully validated RSS link: {rss_link}")
    return rss_link
