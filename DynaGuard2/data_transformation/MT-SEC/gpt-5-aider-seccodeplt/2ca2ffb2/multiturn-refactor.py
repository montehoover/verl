from urllib.parse import urlparse
from typing import Sequence
import logging

logger = logging.getLogger(__name__)

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def is_allowed_scheme(scheme: str, allowed_schemes: Sequence[str] = ALLOWED_SCHEMES) -> bool:
    """
    Pure function that checks if a scheme is allowed.
    """
    if not isinstance(scheme, str) or not scheme:
        return False
    allowed = {s.lower() for s in allowed_schemes}
    return scheme.lower() in allowed


def is_trusted_host(hostname: str, trusted_hosts: Sequence[str] = TRUSTED_IMAGE_HOSTS) -> bool:
    """
    Pure function that checks if a hostname belongs to the trusted hosts.
    """
    if not isinstance(hostname, str) or not hostname:
        return False
    trusted = {h.lower() for h in trusted_hosts}
    return hostname.lower() in trusted


def fetch_image_url(url: str) -> str:
    """
    Validate and return the image URL if it belongs to a trusted image host and uses an allowed scheme.
    Raises ValueError if the URL is invalid, uses an unrecognized scheme, or is not from a trusted host.
    Logs verification attempts, outcomes, and errors.
    """
    logger.info("Verifying image URL: url=%r", url)

    if not isinstance(url, str) or not url.strip():
        logger.error("URL validation failed: input is not a non-empty string: url=%r", url)
        raise ValueError("Invalid URL")

    parsed = urlparse(url.strip())

    if not parsed.scheme or not parsed.netloc:
        logger.error(
            "URL parsing failed: missing scheme or host: url=%r scheme=%r netloc=%r",
            url,
            parsed.scheme,
            parsed.netloc,
        )
        raise ValueError("Invalid URL")

    if not is_allowed_scheme(parsed.scheme, ALLOWED_SCHEMES):
        logger.error(
            "Unrecognized or unsupported URL scheme: scheme=%r url=%r allowed_schemes=%s",
            parsed.scheme,
            url,
            ALLOWED_SCHEMES,
        )
        raise ValueError("Unrecognized or unsupported URL scheme")

    hostname = parsed.hostname or ""
    if not is_trusted_host(hostname, TRUSTED_IMAGE_HOSTS):
        logger.error(
            "Untrusted image host: host=%r url=%r trusted_hosts=%s",
            hostname,
            url,
            TRUSTED_IMAGE_HOSTS,
        )
        raise ValueError("Untrusted image host")

    final_url = parsed.geturl()
    logger.info(
        "Image URL verified successfully: url=%r host=%r scheme=%r",
        final_url,
        hostname.lower(),
        parsed.scheme.lower(),
    )
    return final_url
