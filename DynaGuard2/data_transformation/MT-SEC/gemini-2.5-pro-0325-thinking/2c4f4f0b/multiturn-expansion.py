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
    Checks if a URL is potentially unsafe by matching its hostname against internal IP patterns.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL's hostname matches an internal IP pattern, False otherwise.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    if not hostname:
        return False  # Or True, depending on how you want to handle URLs without hostnames

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False

def check_rss_source(url: str) -> str:
    """
    Verifies if the URL is from a trusted RSS provider and not an internal network.

    Args:
        url: The URL string to check.

    Returns:
        The URL if it's valid and trusted.

    Raises:
        ValueError: If the URL is not trusted or accesses an internal network.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname

    if not hostname:
        raise ValueError("URL must have a hostname.")

    if is_unsafe_url(url):
        raise ValueError(f"URL '{url}' accesses an internal or unsafe network.")

    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL '{url}' is not from a trusted RSS provider.")

    return url

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

if __name__ == '__main__':
    # Example usage:
    example_url = "https://www.example.com/path/to/resource;params?query=value#fragment"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url)
    print(f"URL: {example_url}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")

    # Example usage for check_rss_source:
    trusted_rss_url = "https://rss.trustedsource.com/feed"
    untrusted_rss_url = "https://untrusted.com/feed"
    internal_rss_url = "http://localhost/feed"
    internal_trusted_rss_url = "http://192.168.1.1/feed" # Should be caught by is_unsafe_url

    print(f"\nChecking RSS sources:")
    try:
        print(f"'{trusted_rss_url}' is trusted: {check_rss_source(trusted_rss_url)}")
    except ValueError as e:
        print(f"Error for '{trusted_rss_url}': {e}")

    try:
        check_rss_source(untrusted_rss_url)
    except ValueError as e:
        print(f"Error for '{untrusted_rss_url}': {e}")

    try:
        check_rss_source(internal_rss_url)
    except ValueError as e:
        print(f"Error for '{internal_rss_url}': {e}")

    try:
        # This URL's hostname is not in TRUSTED_RSS_PROVIDERS,
        # but is_unsafe_url should catch it first if it's an internal IP.
        # For this test, let's assume feeds.provider.com is trusted but 192.168.1.1 is not.
        # The current TRUSTED_RSS_PROVIDERS does not include IP addresses.
        check_rss_source(internal_trusted_rss_url)
    except ValueError as e:
        print(f"Error for '{internal_trusted_rss_url}': {e}")

    trusted_provider_url_2 = "http://feeds.provider.com/news"
    try:
        print(f"'{trusted_provider_url_2}' is trusted: {check_rss_source(trusted_provider_url_2)}")
    except ValueError as e:
        print(f"Error for '{trusted_provider_url_2}': {e}")

    # Example usage for is_unsafe_url:
    unsafe_url_1 = "http://localhost:8000/api"
    unsafe_url_2 = "https://192.168.1.100/admin"
    safe_url_1 = "https://www.google.com"
    safe_url_2 = "ftp://example.com/files"
    url_with_no_hostname = "/just/a/path"


    print(f"\nIs '{unsafe_url_1}' unsafe? {is_unsafe_url(unsafe_url_1)}")
    print(f"Is '{unsafe_url_2}' unsafe? {is_unsafe_url(unsafe_url_2)}")
    print(f"Is '{safe_url_1}' unsafe? {is_unsafe_url(safe_url_1)}")
    print(f"Is '{safe_url_2}' unsafe? {is_unsafe_url(safe_url_2)}")
    print(f"Is '{url_with_no_hostname}' unsafe? {is_unsafe_url(url_with_no_hostname)}")

    example_url_2 = "ftp://user:password@host:port/path?query#fragment"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url_2)
    print(f"\nURL: {example_url_2}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")
