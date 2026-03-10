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

def sanitize_feed_url(feed_url: str) -> str:
    """
    Ensures the feed is only fetched from trusted providers, avoiding internal network exposure.

    Args:
        feed_url: str, the RSS feed URL to be sanitized.

    Returns:
        A sanitized and validated RSS feed URL.

    Raises:
        ValueError: If the URL does not belong to trusted providers,
                    attempts to access restricted network URLs,
                    or has an invalid scheme.
    """
    parsed_url = urlparse(feed_url)

    # Validate scheme
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}")

    # Validate hostname
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("URL must have a hostname.")

    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal network URL is restricted: {hostname}")

    # Check against trusted providers
    if hostname not in TRUSTED_RSS_PROVIDERS:
        # Additional check to prevent direct IP access if it's not an internal IP but also not a trusted domain
        # This helps if TRUSTED_RSS_PROVIDERS are domains and we want to block direct IP access in general
        # unless explicitly allowed (which is not the case here based on TRUSTED_RSS_PROVIDERS format)
        is_ip_address = all(c.isdigit() or c == '.' for c in hostname) # Basic IP check
        if is_ip_address:
             # Re-check against internal IP patterns in case the initial check was bypassed by a non-standard format
             # or if we want to be extra cautious about any IP that isn't explicitly trusted.
             # For this problem, if it's an IP and not in TRUSTED_RSS_PROVIDERS (which are domains), it's untrusted.
            raise ValueError(f"Direct IP access is not allowed for untrusted providers: {hostname}")

        raise ValueError(f"URL hostname '{hostname}' is not in the list of trusted providers.")

    return feed_url
