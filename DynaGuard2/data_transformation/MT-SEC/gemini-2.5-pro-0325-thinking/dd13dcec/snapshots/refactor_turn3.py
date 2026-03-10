import re
import logging
from urllib.parse import urlparse

# Configure basic logging
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


def _validate_url_scheme(scheme: str):
    """Validate if the URL scheme is allowed."""
    logger.debug(f"Validating URL scheme: {scheme}")
    if scheme not in ALLOWED_SCHEMES:
        error_msg = (
            f"Invalid URL scheme: {scheme}. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.debug(f"URL scheme '{scheme}' is valid.")


def _is_internal_ip(hostname: str) -> bool:
    """Check if the hostname matches any internal IP patterns."""
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False


def _validate_hostname(hostname: str):
    """Validate the hostname against internal IPs and trusted providers."""
    logger.debug(f"Validating hostname: {hostname}")
    if not hostname:
        error_msg = "URL must have a hostname."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if _is_internal_ip(hostname):
        error_msg = f"Access to internal network URL is restricted: {hostname}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.debug(f"Hostname '{hostname}' is not an internal IP.")

    if hostname not in TRUSTED_RSS_PROVIDERS:
        # Basic IP check (simplified: checks if string could be an IPv4 address)
        # A more robust check might involve trying to parse it as an IP address.
        is_ip_address = all(c.isdigit() or c == '.' for c in hostname) and hostname.count('.') == 3
        if is_ip_address:
            error_msg = f"Direct IP access is not allowed for untrusted providers: {hostname}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        error_msg = f"URL hostname '{hostname}' is not in the list of trusted providers."
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.debug(f"Hostname '{hostname}' is a trusted provider.")


def sanitize_feed_url(feed_url: str) -> str:
    """
    Ensures the feed is only fetched from trusted providers,
    avoiding internal network exposure.

    Args:
        feed_url: str, the RSS feed URL to be sanitized.

    Returns:
        A sanitized and validated RSS feed URL.

    Raises:
        ValueError: If the URL does not belong to trusted providers,
                    attempts to access restricted network URLs,
                    or has an invalid scheme.
    """
    logger.info(f"Attempting to sanitize URL: {feed_url}")
    if not feed_url or not isinstance(feed_url, str):
        error_msg = "Feed URL must be a non-empty string."
        logger.error(f"Validation failed for URL '{feed_url}': {error_msg}")
        raise ValueError(error_msg)

    try:
        parsed_url = urlparse(feed_url)
    except Exception as e: # Catch potential parsing errors from urlparse itself
        error_msg = f"Invalid URL format: {feed_url}. Error: {e}"
        logger.error(f"Validation failed for URL '{feed_url}': {error_msg}")
        raise ValueError(error_msg)

    try:
        _validate_url_scheme(parsed_url.scheme)
        _validate_hostname(parsed_url.hostname)
        logger.info(f"URL successfully sanitized: {feed_url}")
        return feed_url
    except ValueError as ve:
        logger.error(f"Validation failed for URL '{feed_url}': {ve}")
        raise ve
