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

def clean_feed_link(rss_link: str) -> str:
    """
    Ensures an RSS feed URL comes from predefined trusted providers and does not point to internal networks.

    Args:
        rss_link: The RSS feed URL that requires sanitization.

    Returns:
        The sanitized feed URL if valid and trusted.

    Raises:
        ValueError: If the provided URL does not come from an authenticated source,
                    uses an disallowed scheme, or if access to an internal network address is detected.
    """
    try:
        parsed_url = urlparse(rss_link)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {rss_link}. Error: {e}")

    if not parsed_url.scheme or not parsed_url.hostname:
        raise ValueError(f"Invalid URL structure: {rss_link}. Scheme and hostname are required.")

    # Check 1: Validate URL scheme
    if parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: '{parsed_url.scheme}'. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}."
        )

    # Check 2: Prevent access to internal network addresses
    hostname = parsed_url.hostname
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname, re.IGNORECASE):
            raise ValueError(
                f"Access to internal network address detected: '{hostname}'. "
                "This is not allowed."
            )

    # Check 3: Ensure the provider is trusted
    if hostname.lower() not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(
            f"Untrusted RSS provider: '{hostname}'. "
            f"Allowed providers are: {', '.join(TRUSTED_RSS_PROVIDERS)}."
        )

    return rss_link
