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

def check_feed_url(rss_link: str) -> str:
    """
    Validates an RSS feed URL to ensure it comes from a trusted provider
    and does not access internal networks.

    Args:
        rss_link: The RSS feed URL string.

    Returns:
        The sanitized (original) feed URL if it is valid.

    Raises:
        ValueError: If the URL does not meet the security criteria.
    """
    if not isinstance(rss_link, str):
        raise ValueError("Invalid URL type, must be a string.")

    try:
        parsed_url = urlparse(rss_link)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {rss_link}. Error: {e}")

    # Check 1: Scheme must be in ALLOWED_SCHEMES
    if parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Must be one of {ALLOWED_SCHEMES}.")

    # Check 2: Hostname (netloc) must exist
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError(f"URL must have a valid hostname: {rss_link}")
    
    hostname = hostname.lower() # Normalize hostname for checks

    # Check 3: Hostname must not match internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"URL points to an internal network address: {hostname}")

    # Check 4: Hostname must be in TRUSTED_RSS_PROVIDERS
    if hostname not in TRUSTED_RSS_PROVIDERS:
        # Before failing, check if it's an IP address that might have been missed by INTERNAL_IP_PATTERNS
        # (though urlparse.hostname should give a name if it's a name, or IP if it's an IP)
        # This is more of a safeguard; primary check is TRUSTED_RSS_PROVIDERS list.
        # A more robust IP check could use the ipaddress module if needed, but regex covers specified patterns.
        raise ValueError(f"URL hostname '{hostname}' is not a trusted RSS provider.")

    # If all checks pass, return the original (now considered sanitized) URL
    return rss_link

if __name__ == '__main__':
    test_feed_urls = [
        ("http://rss.trustedsource.com/feed", True),
        ("https://feeds.provider.com/news", True),
        ("http://newsfeed.safesite.org/latest", True),
        ("ftp://rss.trustedsource.com/feed", False),  # Bad scheme
        ("http://localhost/feed", False),             # Internal IP (localhost)
        ("https://127.0.0.1/news", False),            # Internal IP (loopback)
        ("http://10.0.0.1/somefeed", False),          # Internal IP (private range)
        ("https://192.168.1.100/updates", False),     # Internal IP (private range)
        ("http://untrusted.com/feed", False),         # Untrusted provider
        ("https://rss.trustedsource.com", True),      # Valid without specific path
        ("http://rss.trustedsource.com:8080/feed", True), # Port is fine
        ("http:///pathonly", False),                  # No hostname
        ("not_a_url", False),                         # Malformed
        ("", False),                                  # Empty string
        (None, False),                                # None input
        ("https://subdomain.rss.trustedsource.com/feed", False) # Subdomain not in trusted list
    ]

    print("Testing check_feed_url function:")
    for url, should_be_valid in test_feed_urls:
        try:
            if url is None and not should_be_valid: # Specific test for None to avoid AttributeError
                 try:
                    check_feed_url(url)
                    print(f"FAIL: {url} -> Expected ValueError, but got success.")
                 except ValueError as e:
                    print(f"PASS: {url} -> Correctly raised ValueError: {e}")
                 except TypeError as e: # Catching type error for None
                    print(f"PASS: {url} -> Correctly raised TypeError/ValueError: {e}")
                 continue

            sanitized_url = check_feed_url(url)
            if should_be_valid:
                print(f"PASS: {url} -> Valid, Sanitized: {sanitized_url}")
            else:
                print(f"FAIL: {url} -> Expected ValueError, but got: {sanitized_url}")
        except ValueError as e:
            if not should_be_valid:
                print(f"PASS: {url} -> Correctly raised ValueError: {e}")
            else:
                print(f"FAIL: {url} -> Expected success, but got ValueError: {e}")
        except Exception as e: # Catch any other unexpected errors
            print(f"ERROR: {url} -> Unexpected error: {e}")

    # Example of a successful call
    print("\nExample of a successful validation:")
    try:
        valid_feed = "https://newsfeed.safesite.org/data"
        result = check_feed_url(valid_feed)
        print(f"Successfully validated: {result}")
    except ValueError as e:
        print(f"Error validating {valid_feed}: {e}")
