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

def filter_rss_feed(rss_feed_url):
    """
    Validates and sanitizes an RSS feed URL.

    Ensures the URL is from a trusted provider, uses an allowed scheme (http/https),
    and does not point to internal network addresses.

    Args:
        rss_feed_url: The RSS feed URL string.

    Returns:
        The sanitized rss_feed_url if it meets all criteria.

    Raises:
        ValueError: If the URL is invalid, not from a trusted provider,
                    uses a disallowed scheme, or points to an internal network.
    """
    if not rss_feed_url or not isinstance(rss_feed_url, str):
        raise ValueError("Invalid URL: Input must be a non-empty string.")

    try:
        parsed_url = urlparse(rss_feed_url)
    except Exception as e: # pylint: disable=broad-except
        raise ValueError(f"Invalid URL format: {rss_feed_url}") from e

    # Check scheme
    if not parsed_url.scheme or parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid scheme: '{parsed_url.scheme}'. Must be one of {ALLOWED_SCHEMES}.")

    # Check hostname
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("Invalid URL: Hostname is missing.")

    hostname_lower = hostname.lower()

    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname_lower):
            raise ValueError(f"Access to internal network address '{hostname}' is not allowed.")

    # Check if hostname is a direct IP address that might have been missed by regex (e.g., non-standard private ranges)
    # This is a basic check; for more robust IP validation, ipaddress module could be used if available/allowed.
    # However, the prompt specified re and urlparse.
    # The current INTERNAL_IP_PATTERNS cover common cases.
    # If it's an IP, and not caught by INTERNAL_IP_PATTERNS, it's assumed public for this function's scope.

    # Check trusted providers
    if hostname_lower not in TRUSTED_RSS_PROVIDERS:
        # Before raising, check if it's an IP address. Trusted providers are usually domain names.
        # If it's an IP address and not an internal one, it's still not a trusted *provider name*.
        is_ip = all(c.isdigit() or c == '.' for c in hostname_lower) # Basic IP check
        if is_ip:
             # Check again with regex for any internal IPs that might have slipped if TRUSTED_RSS_PROVIDERS contained an IP
            is_internal_ip = any(re.match(pattern, hostname_lower) for pattern in INTERNAL_IP_PATTERNS)
            if is_internal_ip:
                raise ValueError(f"Access to internal IP address '{hostname}' is not allowed.")
            # If it's a public IP, it's still not in the trusted provider list by name.
            raise ValueError(f"Hostname '{hostname}' is not a trusted RSS provider. It appears to be a public IP not on the trusted list.")

        raise ValueError(f"Hostname '{hostname}' is not in the list of trusted RSS providers: {TRUSTED_RSS_PROVIDERS}")

    return rss_feed_url

if __name__ == '__main__':
    test_cases = [
        ("http://rss.trustedsource.com/feed", True, "Valid trusted provider"),
        ("https://feeds.provider.com/news", True, "Valid trusted provider with https"),
        ("http://newsfeed.safesite.org/latest.xml", True, "Valid trusted provider"),
        ("ftp://rss.trustedsource.com/feed", False, "Invalid scheme (ftp)"),
        ("http://untrusted.com/feed", False, "Untrusted provider"),
        ("https://localhost/feed", False, "Internal network (localhost)"),
        ("http://127.0.0.1/rss", False, "Internal network (loopback IP)"),
        ("https://10.0.0.5/internalfeed", False, "Internal network (10.x.x.x range)"),
        ("http://192.168.1.10/localnews", False, "Internal network (192.168.x.x range)"),
        ("http://rss.trustedsource.com", True, "Valid trusted provider (no path)"), # Assuming path is not strictly required by filter
        ("https://sub.rss.trustedsource.com/feed", False, "Untrusted (subdomain not explicitly trusted)"),
        ("http://1.2.3.4/feed", False, "Public IP not in trusted list"),
        ("", False, "Empty URL"),
        (None, False, "None URL"),
        ("not_a_url", False, "Malformed URL string"),
        ("http:///path-only", False, "Missing hostname"),
    ]

    print("--- RSS Feed Filter Tests ---")
    for url, should_pass, description in test_cases:
        try:
            result = filter_rss_feed(url)
            if should_pass:
                print(f"PASS: {description} - URL: '{url}' -> '{result}'")
            else:
                print(f"FAIL: {description} - URL: '{url}' - Expected ValueError, but got '{result}'")
        except ValueError as e:
            if should_pass:
                print(f"FAIL: {description} - URL: '{url}' - Expected success, but got ValueError: {e}")
            else:
                print(f"PASS: {description} - URL: '{url}' - Correctly raised ValueError: {e}")
        except Exception as e: # pylint: disable=broad-except
            print(f"UNEXPECTED ERROR: {description} - URL: '{url}' - Error: {e}")

    # Example of a trusted provider that is also an IP (not typical, but for testing logic)
    # To test this, we'd need to add an IP to TRUSTED_RSS_PROVIDERS, e.g. "8.8.8.8"
    # And then test "http://8.8.8.8/feed"
    # Current logic: if it's an IP, it must not be internal. If it's also in TRUSTED_RSS_PROVIDERS, it passes.
    # If it's an IP, not internal, but NOT in TRUSTED_RSS_PROVIDERS, it fails.

    print("\n--- Specific Test for Public IP not in trusted list ---")
    public_ip_url = "http://8.8.8.8/somefeed.xml" # 8.8.8.8 is a public Google DNS server
    try:
        filter_rss_feed(public_ip_url)
        print(f"FAIL: Public IP '{public_ip_url}' should fail as it's not in TRUSTED_RSS_PROVIDERS.")
    except ValueError as e:
        print(f"PASS: Public IP '{public_ip_url}' correctly raised ValueError: {e}")

    print("\n--- Specific Test for Trusted Provider with Port (should pass if hostname matches) ---")
    trusted_with_port = "http://rss.trustedsource.com:8080/feed"
    try:
        result = filter_rss_feed(trusted_with_port)
        print(f"PASS: Trusted with port '{trusted_with_port}' -> '{result}'")
    except ValueError as e:
        print(f"FAIL: Trusted with port '{trusted_with_port}' raised ValueError: {e}")
