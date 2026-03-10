import logging
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TRUSTED_RSS_PROVIDERS = [
    "rss.trustedsource.com",
    "feeds.provider.com",
    "newsfeed.safesite.org",
]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",  # Localhost
    r"^127\.0\.0\.\d+$",  # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$",  # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]


def is_internal_host(host: str, patterns=INTERNAL_IP_PATTERNS) -> bool:
    """
    Determine whether the given host matches internal network patterns.

    Parameters:
        host: The hostname or IP address to evaluate.
        patterns: A list of regex patterns representing internal addresses.

    Returns:
        True if the host matches any internal network pattern, otherwise False.
    """
    if not host:
        return False

    host_lower = host.lower()
    for pattern in patterns:
        if re.fullmatch(pattern, host_lower):
            logger.debug(
                "Internal host detection: host '%s' matched pattern '%s'",
                host_lower,
                pattern,
            )
            return True

    logger.debug("Internal host detection: host '%s' is not internal", host_lower)
    return False


def is_trusted_provider(host: str, providers=TRUSTED_RSS_PROVIDERS) -> bool:
    """
    Check whether the given host is a trusted RSS provider or its subdomain.

    Parameters:
        host: The hostname to validate.
        providers: A list of trusted provider domain names.

    Returns:
        True if the host exactly matches or is a subdomain of a trusted
        provider, otherwise False.
    """
    if not host:
        return False

    host_lower = host.lower()
    for provider in providers:
        provider_lower = provider.lower()
        if host_lower == provider_lower or host_lower.endswith(
            "." + provider_lower
        ):
            logger.debug(
                "Trusted provider check: host '%s' matched '%s'",
                host_lower,
                provider_lower,
            )
            return True

    logger.debug(
        "Trusted provider check: host '%s' did not match any trusted provider",
        host_lower,
    )
    return False


def validate_feed_url(url: str) -> str:
    """
    Validate that the provided RSS feed URL:
      - Uses an allowed scheme (http or https).
      - Does not point to internal network addresses.
      - Belongs to a trusted RSS provider (exact domain or its subdomain).

    Parameters:
        url: The RSS feed URL to validate.

    Returns:
        The original URL if it passes validation.

    Raises:
        ValueError: If the URL is empty/invalid, uses a disallowed scheme,
                    targets an internal network address, or is not from a
                    trusted provider.
    """
    logger.info("Starting feed URL validation: url='%s'", url)

    if not isinstance(url, str) or not url.strip():
        logger.warning("Validation failed: URL must be a non-empty string")
        raise ValueError("A non-empty URL string is required.")

    url = url.strip()
    parsed = urlparse(url)

    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    logger.debug("Parsed URL components: scheme='%s', host='%s'", scheme, host)

    if scheme not in ALLOWED_SCHEMES:
        logger.warning(
            "Validation failed: disallowed scheme '%s' for url '%s'", scheme, url
        )
        raise ValueError("URL scheme must be http or https.")

    if not host:
        logger.warning("Validation failed: missing host for url '%s'", url)
        raise ValueError("URL must include a valid host.")

    if is_internal_host(host):
        logger.warning(
            "Validation failed: internal network access detected for host '%s' "
            "in url '%s'",
            host,
            url,
        )
        raise ValueError("Access to internal network addresses is not allowed.")

    if not is_trusted_provider(host):
        logger.warning(
            "Validation failed: untrusted RSS provider '%s' for url '%s'",
            host,
            url,
        )
        raise ValueError("URL is not from a trusted RSS provider.")

    logger.info(
        "Validation successful: url '%s' is allowed (host='%s', scheme='%s')",
        url,
        host,
        scheme,
    )
    return url
