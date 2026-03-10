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

def check_rss_source(source_url: str) -> str:
    """
    Verifies an RSS feed URL is from a trusted provider and not accessing internal networks.

    Args:
        source_url: The URL string of the RSS feed.

    Returns:
        The source_url if it is valid and trusted.

    Raises:
        ValueError: If the URL does not meet the criteria (invalid scheme,
                    not a trusted provider, or targets an internal network).
    """
    parsed_url = urlparse(source_url)

    # Check scheme
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: '{parsed_url.scheme}'. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}."
        )

    # Check if netloc is a trusted provider
    if parsed_url.netloc not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(
            f"URL domain '{parsed_url.netloc}' is not a trusted RSS provider. "
            f"Trusted providers are: {', '.join(TRUSTED_RSS_PROVIDERS)}."
        )

    # Check for internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, parsed_url.netloc):
            raise ValueError(
                f"URL domain '{parsed_url.netloc}' matches an internal IP pattern "
                f"and is not allowed."
            )

    return source_url
