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

def validate_rss_link(rss_link: str) -> str:
    """
    Validates an RSS feed URL against trusted providers and internal network checks.

    Args:
        rss_link: The RSS feed URL string.

    Returns:
        The sanitized feed URL if it is valid.

    Raises:
        ValueError: If the URL is invalid, not from a trusted provider,
                    uses a disallowed scheme, or points to an internal IP.
    """
    try:
        parsed_url = urlparse(rss_link)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {rss_link}") from e

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL structure: {rss_link}. Scheme and domain must be present.")

    if parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(f"Disallowed scheme: {parsed_url.scheme}. Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}")

    domain = parsed_url.hostname  # Use hostname for IP checks and provider matching
    if not domain:
        raise ValueError(f"Could not extract domain from URL: {rss_link}")

    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, domain):
            raise ValueError(f"Access to internal network resource is forbidden: {domain}")

    # Check if the domain is a trusted RSS provider
    if domain.lower() not in TRUSTED_RSS_PROVIDERS:
        # Allow subdomains of trusted providers if the main domain is listed
        # e.g. if "provider.com" is trusted, "sub.provider.com" should also be trusted.
        is_trusted_subdomain = False
        for trusted_provider in TRUSTED_RSS_PROVIDERS:
            if domain.lower().endswith(f".{trusted_provider}") or domain.lower() == trusted_provider:
                is_trusted_subdomain = True
                break
        if not is_trusted_subdomain:
            raise ValueError(f"Domain not trusted: {domain}. Must be one of {', '.join(TRUSTED_RSS_PROVIDERS)}")

    # Return the sanitized URL (scheme and netloc in lowercase, rest as is)
    # This helps in normalizing URLs but keeps path and query params case-sensitive if needed.
    sanitized_url = parsed_url._replace(scheme=parsed_url.scheme.lower(), netloc=parsed_url.netloc.lower()).geturl()
    return sanitized_url
