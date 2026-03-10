import logging
import re
from urllib.parse import urlparse, urlunparse

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


logger = logging.getLogger(__name__)


def is_internal_hostname(hostname: str, patterns=INTERNAL_IP_PATTERNS) -> bool:
    """
    Return True if the hostname matches any internal or loopback pattern.
    """
    for pattern in patterns:
        if re.match(pattern, hostname, flags=re.IGNORECASE):
            return True
    return False


def is_trusted_provider(hostname: str, providers=TRUSTED_RSS_PROVIDERS) -> bool:
    """
    Return True if the hostname exactly matches one of the trusted providers.
    """
    trusted_hosts = {h.lower() for h in providers}
    return hostname.lower() in trusted_hosts


def validate_rss_link(rss_link: str) -> str:
    """
    Validate and sanitize an RSS feed URL, ensuring it points only to trusted providers
    and does not attempt to access restricted internal networks.

    This function logs each validation attempt, including the raw input, whether
    validation succeeded, and any errors encountered.

    Args:
        rss_link: The URL of the RSS feed being validated.

    Returns:
        A sanitized feed URL limited to trusted providers.

    Raises:
        ValueError: If the URL is invalid, uses a disallowed scheme, is not from an
                    approved provider, or attempts access to restricted internal networks.
    """
    logger.debug("RSS validation attempt started: rss_link=%r", rss_link)

    try:
        if not isinstance(rss_link, str):
            raise ValueError("RSS link must be a string.")

        link = rss_link.strip()
        if not link:
            raise ValueError("RSS link cannot be empty.")

        parsed = urlparse(link)

        # Validate scheme.
        scheme = (parsed.scheme or "").lower()
        if scheme not in ALLOWED_SCHEMES:
            raise ValueError(f"URL scheme must be one of {ALLOWED_SCHEMES}.")

        # Extract and normalize hostname (strip trailing dot, lowercase).
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if not hostname:
            raise ValueError("URL must include a valid hostname.")

        # Block obvious internal targets.
        if is_internal_hostname(hostname):
            raise ValueError("Access to internal or loopback addresses is not allowed.")

        # Enforce trusted providers (exact host match).
        if not is_trusted_provider(hostname):
            raise ValueError("URL host is not in the list of approved RSS providers.")

        # Validate and normalize port.
        port = parsed.port
        if port is not None:
            # Only allow default ports for the given scheme.
            if (scheme == "http" and port != 80) or (scheme == "https" and port != 443):
                raise ValueError("Non-standard ports are not allowed for RSS feeds.")
            # Default ports can be omitted in the sanitized result.
            port = None

        # Reconstruct sanitized netloc: no userinfo, lowercase host, optional port.
        netloc = hostname if port is None else f"{hostname}:{port}"

        # Normalize path (keep as-is; default to '/' if empty).
        path = parsed.path or "/"

        # Keep params and query; drop fragment.
        sanitized_url = urlunparse(
            (
                scheme,
                netloc,
                path,
                parsed.params,
                parsed.query,
                "",  # drop fragment
            )
        )

        logger.info(
            "RSS validation succeeded: rss_link=%r sanitized_url=%r",
            rss_link,
            sanitized_url,
        )
        return sanitized_url

    except ValueError as exc:
        logger.warning(
            "RSS validation failed: rss_link=%r error=%s",
            rss_link,
            str(exc),
        )
        raise

    except Exception as exc:
        logger.exception(
            "Unexpected error during RSS validation: rss_link=%r", rss_link
        )
        raise ValueError("Invalid RSS link.") from exc
