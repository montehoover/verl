import re
from urllib.parse import urlparse

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]

def is_unsafe_url(url: str) -> bool:
    """
    Checks if a URL points to an internal or potentially unsafe IP address.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL is considered unsafe, False otherwise.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname  # Use .hostname to get just the host, without port

    if not hostname:
        return False  # Cannot determine hostname, assume not unsafe for this check

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False

def extract_url_parts(url: str) -> tuple[str, str, str, str, str, str]:
    """
    Extracts the components of a URL.

    Args:
        url: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, path, params, query, and fragment.
    """
    parsed_url = urlparse(url)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )

def clean_feed_link(feed_url: str) -> str:
    """
    Validates and sanitizes an RSS feed URL.

    Checks if the URL is from a trusted provider and does not point to an internal network.

    Args:
        feed_url: The RSS feed URL to validate.

    Returns:
        The sanitized URL if it's valid and trusted.

    Raises:
        ValueError: If the URL is not trusted, invalid, or points to an internal network.
    """
    parsed_url = urlparse(feed_url)

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL format: {feed_url}")

    if is_unsafe_url(feed_url):
        raise ValueError(f"URL points to an internal or unsafe network: {feed_url}")

    # .hostname extracts the host part, ignoring port if present
    hostname = parsed_url.hostname
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL is not from a trusted provider: {feed_url}")

    return feed_url

if __name__ == '__main__':
    # Example usage:
    example_url = "http://user:password@www.example.com:80/path/to/resource;params?query=value#fragment"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url)
    print(f"URL: {example_url}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")

    # Test cases for clean_feed_link
    print("\nTesting clean_feed_link:")
    trusted_feed_urls = [
        "http://rss.trustedsource.com/feed",
        "https://feeds.provider.com/news",
        "http://newsfeed.safesite.org/updates?format=xml"
    ]
    untrusted_feed_urls = [
        "http://untrusted.com/feed",
        "https://localhost/myfeed",
        "http://192.168.1.1/rss",
        "ftp://rss.trustedsource.com/feed" # Potentially valid if ftp was allowed, but scheme check is basic
    ]

    print("\n  Trusted feed URLs (should pass):")
    for t_url in trusted_feed_urls:
        try:
            cleaned_url = clean_feed_link(t_url)
            print(f"    Original: {t_url}, Cleaned: {cleaned_url} - OK")
        except ValueError as e:
            print(f"    Original: {t_url}, Error: {e} - FAILED (unexpected)")

    print("\n  Untrusted/Unsafe feed URLs (should fail):")
    for ut_url in untrusted_feed_urls:
        try:
            clean_feed_link(ut_url)
            print(f"    Original: {ut_url} - FAILED (should have raised ValueError)")
        except ValueError as e:
            print(f"    Original: {ut_url}, Error: {e} - OK")

    # Example of an invalid URL format
    invalid_format_url = "not_a_url"
    print(f"\n  Invalid format URL (should fail):")
    try:
        clean_feed_link(invalid_format_url)
        print(f"    Original: {invalid_format_url} - FAILED (should have raised ValueError)")
    except ValueError as e:
        print(f"    Original: {invalid_format_url}, Error: {e} - OK")
