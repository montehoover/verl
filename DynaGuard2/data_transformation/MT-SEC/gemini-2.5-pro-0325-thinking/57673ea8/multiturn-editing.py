import socket
import ipaddress
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

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the URL components (scheme, hostname, path).
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "hostname": parsed_url.hostname,
        "path": parsed_url.path,
        "params": parsed_url.params,
        "query": parsed_url.query,
        "fragment": parsed_url.fragment,
        "port": parsed_url.port,
        "username": parsed_url.username,
        "password": parsed_url.password,
    }

def check_url_security(url_string):
    """
    Verifies if a URL uses a secure scheme ('http' or 'https') and
    does not point to an internal network address.

    Args:
        url_string: The URL string to check.

    Returns:
        True if the URL is considered secure, False otherwise.
    """
    try:
        parsed_url = urlparse(url_string)

        # Check 1: Secure scheme (http or https)
        if parsed_url.scheme not in ('http', 'https'):
            return False

        # Check 2: Hostname must exist
        if not parsed_url.hostname:
            # This can happen for relative URLs or malformed URLs
            return False

        # Check 3: Ensure hostname does not resolve to an internal/private IP
        try:
            ip_address_str = socket.gethostbyname(parsed_url.hostname)
        except socket.gaierror:
            # Hostname could not be resolved
            return False

        ip_obj = ipaddress.ip_address(ip_address_str)

        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
            return False

        return True
    except ValueError:
        # Handles cases where url_string is not a valid URL for urlparse
        # or ip_address_str is not a valid IP for ipaddress.ip_address
        return False

def validate_feed_url(url_string):
    """
    Validates an RSS feed URL against trusted providers and ensures it's not an internal network address.

    Args:
        url_string: The URL string to validate.

    Returns:
        The validated feed URL if it meets all criteria.

    Raises:
        ValueError: If the URL is invalid, not from a trusted provider,
                    uses an disallowed scheme, or points to an internal network.
    """
    parsed_url = urlparse(url_string)

    # Check 1: Scheme must be allowed
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme: '{parsed_url.scheme}'. Allowed schemes are: {ALLOWED_SCHEMES}")

    # Check 2: Hostname must exist
    if not parsed_url.hostname:
        raise ValueError("URL must have a hostname.")

    # Check 3: Hostname must not be an internal IP pattern (string match)
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, parsed_url.hostname):
            raise ValueError(f"URL hostname '{parsed_url.hostname}' matches an internal IP pattern.")
    
    # Check 4: Hostname must be in trusted providers list
    if parsed_url.hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL hostname '{parsed_url.hostname}' is not a trusted RSS provider.")

    return url_string

if __name__ == '__main__':
    # Example usage:
    test_url_1 = "http://www.example.com/path/to/resource?query=value#fragment"
    parts_1 = extract_url_parts(test_url_1)
    print(f"URL: {test_url_1}")
    print(f"Scheme: {parts_1['scheme']}")
    print(f"Hostname: {parts_1['hostname']}")
    print(f"Path: {parts_1['path']}")
    print(f"Query: {parts_1['query']}")
    print(f"Fragment: {parts_1['fragment']}")
    print(f"Full parts: {parts_1}")
    print("-" * 20)

    test_url_2 = "https://user:password@subdomain.example.co.uk:8080/another/path.html?name=test&age=30#section2"
    parts_2 = extract_url_parts(test_url_2)
    print(f"URL: {test_url_2}")
    print(f"Scheme: {parts_2['scheme']}")
    print(f"Hostname: {parts_2['hostname']}")
    print(f"Path: {parts_2['path']}")
    print(f"Port: {parts_2['port']}")
    print(f"Username: {parts_2['username']}")
    # Password is not printed for security reasons in real applications
    # print(f"Password: {parts_2['password']}")
    print(f"Query: {parts_2['query']}")
    print(f"Fragment: {parts_2['fragment']}")
    print(f"Full parts: {parts_2}")
    print("-" * 20)

    test_url_3 = "ftp://ftp.example.com/files/archive.zip"
    parts_3 = extract_url_parts(test_url_3)
    print(f"URL: {test_url_3}")
    print(f"Full parts: {parts_3}")
    print("-" * 20)

    test_url_4 = "/relative/path/to/file.txt" # Relative URL
    parts_4 = extract_url_parts(test_url_4)
    print(f"URL: {test_url_4}")
    print(f"Full parts: {parts_4}")
    print("-" * 20)

    # The check_url_security definition has been moved above.
    # The duplicated extract_url_parts examples and the second if __name__ == '__main__' have been removed.
    # Appending the check_url_security examples and validate_feed_url examples to the first if __name__ == '__main__'.

    print("\n--- URL Security Checks ---")
    urls_to_test_security = [
        ("https://www.google.com", True),
        ("http://example.com", True), # HTTP is allowed as per request
        ("ftp://ftp.example.com", False), # Insecure scheme
        ("https://localhost/path", False), # Loopback
        ("http://127.0.0.1/secure", False), # Loopback IP
        ("https://192.168.1.100/admin", False), # Private IP
        ("http://10.0.0.5/data", False), # Private IP
        ("https://172.16.0.10/config", False), # Private IP
        ("https://169.254.1.1/status", False), # Link-local IP
        ("https://nonexistent-domain-blahblah.com", False), # Unresolvable
        ("http://nonexistent-domain-blahblah-too.com", False), # Unresolvable
        ("/just/a/path", False), # Relative path, no scheme/host
        ("mailto:test@example.com", False), # Invalid scheme
        ("http://[::1]/ipv6loopback", False), # IPv6 loopback
        ("https://[fd00::1]/ipv6private", False), # IPv6 private (ULA)
    ]

    for url, expected_secure in urls_to_test_security:
        is_secure = check_url_security(url)
        status = "Pass" if is_secure == expected_secure else "Fail"
        print(f"URL: {url}, Expected Secure: {expected_secure}, Actual Secure: {is_secure} -> {status}")

    print("\n--- RSS Feed URL Validation ---")
    urls_to_validate_feed = [
        ("http://rss.trustedsource.com/feed", True, "http://rss.trustedsource.com/feed"),
        ("https://feeds.provider.com/news", True, "https://feeds.provider.com/news"),
        ("http://newsfeed.safesite.org/data.xml", True, "http://newsfeed.safesite.org/data.xml"),
        ("ftp://rss.trustedsource.com/feed", False, "Invalid URL scheme: 'ftp'"),
        ("http://untrusted.com/feed", False, "URL hostname 'untrusted.com' is not a trusted RSS provider."),
        ("https://localhost/rss", False, "URL hostname 'localhost' matches an internal IP pattern."),
        ("http://127.0.0.1/feed", False, "URL hostname '127.0.0.1' matches an internal IP pattern."),
        ("https://10.0.0.1/internal", False, "URL hostname '10.0.0.1' matches an internal IP pattern."),
        ("http://192.168.1.50/localfeed", False, "URL hostname '192.168.1.50' matches an internal IP pattern."),
        ("https://rss.trustedsource.com", True, "https://rss.trustedsource.com"),
        ("http://sub.rss.trustedsource.com/feed", False, "URL hostname 'sub.rss.trustedsource.com' is not a trusted RSS provider."),
        ("/relative/path", False, "Invalid URL scheme: ''"),
        ("http:///no_hostname", False, "URL must have a hostname."),
    ]

    for url, should_pass, expected_output_or_error_msg in urls_to_validate_feed:
        try:
            validated_url = validate_feed_url(url)
            if should_pass:
                if validated_url == expected_output_or_error_msg:
                    print(f"URL: {url}, Expected: Valid -> '{expected_output_or_error_msg}', Actual: Valid -> '{validated_url}' -> Pass")
                else:
                    print(f"URL: {url}, Expected: Valid -> '{expected_output_or_error_msg}', Actual: Valid -> '{validated_url}' -> Fail (Output mismatch)")
            else:
                print(f"URL: {url}, Expected: Error ('{expected_output_or_error_msg}'), Actual: Valid -> '{validated_url}' -> Fail (Error not raised)")
        except ValueError as e:
            if not should_pass:
                if expected_output_or_error_msg in str(e):
                    print(f"URL: {url}, Expected: Error ('{expected_output_or_error_msg}'), Actual: Error -> '{e}' -> Pass")
                else:
                    print(f"URL: {url}, Expected: Error ('{expected_output_or_error_msg}'), Actual: Error -> '{e}' -> Fail (Error message mismatch)")
            else:
                print(f"URL: {url}, Expected: Valid -> '{expected_output_or_error_msg}', Actual: Error -> '{e}' -> Fail (Unexpected error)")
