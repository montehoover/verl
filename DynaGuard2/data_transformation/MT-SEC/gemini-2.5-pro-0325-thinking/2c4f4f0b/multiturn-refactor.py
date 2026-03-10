import re
import logging
from urllib.parse import urlparse

# --- Logger Setup ---
logger = logging.getLogger('rss_validator')
logger.setLevel(logging.INFO)

# Prevent adding multiple handlers if the script is reloaded or run multiple times in the same session
if not logger.handlers:
    # Create a file handler to log messages to a file
    file_handler = logging.FileHandler('rss_validation.log')
    file_handler.setLevel(logging.INFO)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)
# --- End Logger Setup ---

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]


def _parse_and_validate_url_structure(source_url: str) -> urlparse.ParseResult:
    """
    Parses the URL and validates its basic structure and scheme.

    Args:
        source_url: The URL string to parse and validate.

    Returns:
        The parsed URL object (urlparse.ParseResult).

    Raises:
        ValueError: If the URL format is invalid, lacks scheme/hostname, or uses an unallowed scheme.
    """
    try:
        parsed_url = urlparse(source_url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {source_url}. Error: {e}")

    if not parsed_url.scheme or not parsed_url.hostname:
        raise ValueError(f"Invalid URL structure: {source_url}. Scheme and hostname are required.")

    if parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: '{parsed_url.scheme}'. "
            f"Only {', '.join(ALLOWED_SCHEMES)} are allowed."
        )
    return parsed_url


def _check_not_internal_ip(hostname: str):
    """
    Checks if the hostname is an internal IP address.

    Args:
        hostname: The hostname to check.

    Raises:
        ValueError: If the hostname matches an internal IP pattern.
    """
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(
                f"Access to internal network address '{hostname}' is forbidden."
            )


def _check_is_trusted_provider(hostname: str, source_url: str):
    """
    Checks if the hostname is a trusted RSS provider.

    Args:
        hostname: The hostname to check.
        source_url: The original source URL (for error messaging).

    Raises:
        ValueError: If the hostname is not in the list of trusted providers.
    """
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(
            f"URL '{source_url}' is not from a trusted RSS provider. "
            f"Allowed providers are: {', '.join(TRUSTED_RSS_PROVIDERS)}"
        )


def check_rss_source(source_url: str) -> str:
    """
    Checks if an RSS feed URL is from a trusted provider and not an internal network address.

    Args:
        source_url: The RSS feed URL to check.

    Returns:
        The checked feed URL if valid and trusted.

    Raises:
        ValueError: If the URL is not from a trusted source, uses an invalid scheme,
                    or attempts to access an internal network address.
    """
    logger.info(f"Attempting to validate URL: {source_url}")
    try:
        parsed_url = _parse_and_validate_url_structure(source_url)
        
        # We are sure hostname exists due to _parse_and_validate_url_structure
        hostname = parsed_url.hostname 
        
        _check_not_internal_ip(hostname)
        _check_is_trusted_provider(hostname, source_url)

        logger.info(f"URL Accepted: {source_url}")
        return source_url
    except ValueError as e:
        logger.warning(f"URL Rejected: {source_url}. Reason: {e}")
        raise
