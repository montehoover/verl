import re
from urllib.parse import urlparse

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]

def extract_url_parts(url_string):
    """
    Parses a URL string and returns its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, path, params, query, and fragment.
    """
    parsed_url = urlparse(url_string)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )

def is_internal_network_url(url_string):
    """
    Checks if the given URL points to an internal network based on predefined patterns.

    Args:
        url_string: The URL string to check.

    Returns:
        True if the URL's hostname matches an internal network pattern, False otherwise.
    """
    parsed_url = urlparse(url_string)
    hostname = parsed_url.hostname  # Use .hostname to ignore port if present

    if not hostname:
        return False

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False

def check_feed_url(feed_url):
    """
    Verifies an RSS feed URL against trusted providers and checks for restricted network access.

    Args:
        feed_url: The RSS feed URL string to check.

    Returns:
        The sanitized URL if valid.

    Raises:
        ValueError: If the URL is not trusted or attempts to connect to a restricted network.
    """
    parsed_url = urlparse(feed_url)
    hostname = parsed_url.hostname

    if not hostname:
        raise ValueError("Invalid URL: No hostname found.")

    if is_internal_network_url(feed_url):
        raise ValueError(f"Restricted network access: URL '{feed_url}' points to an internal network.")

    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"Untrusted RSS provider: Hostname '{hostname}' is not in the trusted list.")

    # If all checks pass, return the original URL (considered "sanitized" by validation)
    return feed_url

if __name__ == '__main__':
    # Example usage for extract_url_parts:
    example_url = "http://user:password@www.example.com:80/path/to/resource;params?query=value#fragment"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url)
    print(f"URL: {example_url}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")

    # Example usage for check_feed_url:
    print("\nFeed URL Check:")
    test_feed_urls = [
        "http://rss.trustedsource.com/feed",
        "https://feeds.provider.com/news",
        "http://newsfeed.safesite.org/updates",
        "http://untrusted.com/feed",
        "http://localhost/myfeed",
        "http://10.0.0.1/internalrss",
        "http://192.168.1.50/localnews",
        "https://rss.trustedsource.com:8080/feed" # Trusted host with port
    ]

    for url_to_check in test_feed_urls:
        try:
            sanitized_url = check_feed_url(url_to_check)
            print(f"  URL: {url_to_check} -> Valid: {sanitized_url}")
        except ValueError as e:
            print(f"  URL: {url_to_check} -> Invalid: {e}")

    # Example usage for is_internal_network_url:
    urls_to_test = [
        "http://localhost/admin",
        "http://127.0.0.1/dashboard",
        "http://10.0.0.5/internal_service",
        "http://192.168.1.100/config",
        "http://www.google.com",
        "ftp://10.1.2.3/files",
        "http://172.16.0.1/test", # Should be False based on current patterns
        "http://example.com",
        "http://192.168.1.256", # Invalid IP, but pattern might partially match hostname
        "http://localhost:8080/api"
    ]

    print("\nInternal Network URL Check:")
    for url_test in urls_to_test:
        is_internal = is_internal_network_url(url_test)
        print(f"  URL: {url_test} -> Is Internal: {is_internal}")

    example_url_2 = "ftp://ftp.example.com/files/archive.zip"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url_2)
    print(f"\nURL: {example_url_2}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")
