import re
import logging
from urllib.parse import urlparse

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]

# Configure logger to write to file in the current directory
LOGGER_NAME = "rss_validation"
LOG_FILE = "rss_validation.log"
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def validate_url(source_url: str) -> str:
    """
    Validate the RSS feed URL for correct scheme, presence of hostname, and
    protection against internal network targets. Returns the normalized hostname.

    Args:
        source_url: The RSS feed URL to validate.

    Returns:
        The normalized hostname extracted from the URL.

    Raises:
        ValueError: If the URL is empty/invalid, uses a disallowed scheme,
        lacks a valid hostname, or targets an internal network address.
    """
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("A non-empty RSS feed URL string is required.")

    parsed = urlparse(source_url.strip())

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError("URL scheme not allowed. Only http and https are permitted.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid hostname.")

    host = hostname.lower().strip(".")

    # Block access to internal network addresses
    for pattern in INTERNAL_IP_PATTERNS:
        if re.fullmatch(pattern, host):
            raise ValueError("Access to internal network addresses is not allowed.")

    return host


def is_trusted_provider(host: str) -> bool:
    """
    Check whether the provided hostname is part of the trusted RSS providers.
    """
    return host in TRUSTED_RSS_PROVIDERS


def check_rss_source(source_url: str) -> str:
    """
    Validate that the given RSS feed URL uses an allowed scheme, does not target
    an internal network address, and belongs to a trusted provider.

    Args:
        source_url: The RSS feed URL to validate.

    Returns:
        The original URL if it is valid and trusted.

    Raises:
        ValueError: If the URL is invalid, uses a disallowed scheme, targets an
        internal network address, or is not from a trusted provider.
    """
    logger.info("Checking RSS source URL: %s", source_url)
    try:
        host = validate_url(source_url)

        if not is_trusted_provider(host):
            raise ValueError(f"The provided RSS feed URL is not from a trusted provider: host={host}")

        logger.info("URL accepted: url=%s host=%s", source_url, host)
        return source_url
    except ValueError as e:
        logger.warning("URL rejected: url=%s reason=%s", source_url, str(e))
        raise
