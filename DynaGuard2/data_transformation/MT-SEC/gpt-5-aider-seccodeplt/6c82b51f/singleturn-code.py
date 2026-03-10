import re
from urllib.parse import urlparse, urlunparse

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]


def check_feed_url(rss_link: str) -> str:
    """
    Validate and sanitize an RSS feed URL.

    - Only allow http/https schemes.
    - Only allow hosts from TRUSTED_RSS_PROVIDERS.
    - Disallow access to internal network addresses based on INTERNAL_IP_PATTERNS.
    - Disallow credentials in the URL.
    - Return a sanitized URL with normalized scheme/hostname.
    - Raise ValueError if validation fails.

    Args:
        rss_link: RSS feed URL string to validate.

    Returns:
        Sanitized RSS feed URL string.

    Raises:
        ValueError: If the URL is invalid, from an untrusted provider,
                    or targets a restricted/internal network.
    """
    if not isinstance(rss_link, str):
        raise ValueError("RSS link must be a string.")

    rss_link = rss_link.strip()
    if not rss_link:
        raise ValueError("RSS link cannot be empty.")

    parsed = urlparse(rss_link)

    # Validate scheme
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("Unsupported or missing URL scheme. Allowed schemes are http and https.")

    # Validate host/netloc existence
    if not parsed.netloc or not parsed.hostname:
        raise ValueError("URL must include a valid host.")

    # Disallow embedded credentials
    if parsed.username or parsed.password:
        raise ValueError("Credentials in URL are not allowed.")

    hostname = parsed.hostname.lower()

    # Deny internal network targets using provided patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError("Access to internal network addresses is not allowed.")

    # Restrict to trusted providers (exact hostname match)
    trusted = any(hostname == provider.lower() for provider in TRUSTED_RSS_PROVIDERS)
    if not trusted:
        raise ValueError("URL host is not a trusted RSS provider.")

    # Sanitize: normalize scheme and hostname, drop any credentials if present (already checked)
    # Preserve port, path, params, query, fragment.
    netloc = hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    sanitized = urlunparse((
        scheme,
        netloc,
        parsed.path or "",
        parsed.params or "",
        parsed.query or "",
        parsed.fragment or "",
    ))

    return sanitized
