import re
from typing import Iterable, Pattern, Sequence
from urllib.parse import ParseResult, urlparse, urlunparse

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


# Pre-compile internal IP regex patterns for performance and clarity.
INTERNAL_IP_REGEXES: Sequence[Pattern[str]] = [
    re.compile(p) for p in INTERNAL_IP_PATTERNS
]


def is_internal_host(host: str, patterns: Iterable[Pattern[str]] = INTERNAL_IP_REGEXES) -> bool:
    """
    Determine if a hostname resolves to an internal or restricted network address.

    A host is considered internal if it matches any of the provided regex patterns.

    Args:
        host: Hostname or IP address (string) to evaluate.
        patterns: Iterable of compiled regex patterns that indicate internal hosts.

    Returns:
        True if the host matches an internal pattern, otherwise False.
    """
    if not host:
        return False

    for regex in patterns:
        if regex.match(host):
            return True

    return False


def is_trusted_provider(host: str, providers: Sequence[str] = TRUSTED_RSS_PROVIDERS) -> bool:
    """
    Check if the given host belongs to the trusted RSS providers allowlist.

    Matches are allowed for exact provider domains or any of their subdomains.

    Args:
        host: Hostname to check (will be treated case-insensitively).
        providers: A sequence of trusted provider domain names.

    Returns:
        True if the host is trusted, otherwise False.
    """
    if not host:
        return False

    host = host.lower()

    for provider in providers:
        provider = provider.lower()
        if host == provider or host.endswith("." + provider):
            return True

    return False


def _sanitize_parsed_url(parsed: ParseResult) -> str:
    """
    Construct a sanitized URL string from a parsed URL, enforcing:
      - lowercase scheme and host
      - no fragments
      - no credentials in netloc (assumed validated separately)

    Args:
        parsed: The parsed URL (ParseResult).

    Returns:
        A sanitized URL string.
    """
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()

    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"

    path = parsed.path or ""
    params = parsed.params or ""
    query = parsed.query or ""
    fragment = ""  # Drop any fragment

    return urlunparse((scheme, netloc, path, params, query, fragment))


def check_feed_url(rss_link: str) -> str:
    """
    Validate and sanitize an RSS feed URL.

    Enforces:
      - Allowed schemes (http/https).
      - No embedded credentials.
      - Host not targeting internal networks.
      - Host belongs to trusted providers (including their subdomains).

    Args:
        rss_link: The RSS feed URL string to validate.

    Returns:
        A sanitized, valid RSS feed URL string.

    Raises:
        ValueError: If validation fails for any rule.
    """
    if not isinstance(rss_link, str) or not rss_link.strip():
        raise ValueError("RSS link must be a non-empty string")

    parsed = urlparse(rss_link.strip())

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("URL scheme is not allowed")

    # Disallow embedded credentials for security.
    if parsed.username or parsed.password:
        raise ValueError("Credentials in URL are not allowed")

    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("URL must include a valid host")

    # Block internal/restricted network targets.
    if is_internal_host(host):
        raise ValueError("Access to internal network addresses is not allowed")

    # Allow only trusted providers (including their subdomains).
    if not is_trusted_provider(host):
        raise ValueError("URL is not from a trusted RSS provider")

    # Rebuild a sanitized URL and return it.
    return _sanitize_parsed_url(parsed)
