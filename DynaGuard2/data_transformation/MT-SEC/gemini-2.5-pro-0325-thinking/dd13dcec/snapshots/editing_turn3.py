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
        A dictionary containing the scheme, hostname, and path of the URL.
        Returns None for hostname or path if they are not present.
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "hostname": parsed_url.hostname,
        "path": parsed_url.path
    }

def is_trusted_domain(url_string, trusted_domains):
    """
    Verifies if the URL's hostname is part of a predefined list of trusted domains.

    Args:
        url_string: The URL string to check.
        trusted_domains: A list or set of trusted domain names.

    Returns:
        True if the URL's hostname is in the trusted_domains list, False otherwise.
        Returns False if the URL has no hostname.
    """
    parts = extract_url_parts(url_string)
    hostname = parts.get("hostname")
    if hostname:
        return hostname in trusted_domains
    return False

def sanitize_feed_url(feed_url):
    """
    Sanitizes and validates an RSS feed URL.

    Ensures the URL is from a trusted provider, uses an allowed scheme,
    and does not point to an internal network resource.

    Args:
        feed_url: The RSS feed URL string.

    Returns:
        The sanitized and validated RSS feed URL.

    Raises:
        ValueError: If the URL is invalid, not from a trusted provider,
                    uses a disallowed scheme, or points to an internal IP.
    """
    parsed_url = urlparse(feed_url)

    # Check scheme
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Allowed schemes are: {ALLOWED_SCHEMES}")

    # Check hostname presence
    if not parsed_url.hostname:
        raise ValueError("URL must have a hostname.")

    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, parsed_url.hostname):
            raise ValueError(f"Hostname '{parsed_url.hostname}' matches an internal IP pattern.")

    # Check if hostname is in trusted providers
    if parsed_url.hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"Hostname '{parsed_url.hostname}' is not a trusted RSS provider.")

    return feed_url

if __name__ == '__main__':
    # Example usage for extract_url_parts:
    test_url_1 = "http://www.example.com/path/to/page?query=string#fragment"
    parts_1 = extract_url_parts(test_url_1)
    print(f"URL: {test_url_1}")
    print(f"Scheme: {parts_1['scheme']}")
    print(f"Hostname: {parts_1['hostname']}")
    print(f"Path: {parts_1['path']}")
    print("-" * 20)

    test_url_2 = "https://subdomain.example.co.uk:8080/another/path.html"
    parts_2 = extract_url_parts(test_url_2)
    print(f"URL: {test_url_2}")
    print(f"Scheme: {parts_2['scheme']}")
    print(f"Hostname: {parts_2['hostname']}")
    print(f"Path: {parts_2['path']}")
    print("-" * 20)

    test_url_3 = "ftp://ftp.example.org/resource"
    parts_3 = extract_url_parts(test_url_3)
    print(f"URL: {test_url_3}")
    print(f"Scheme: {parts_3['scheme']}")
    print(f"Hostname: {parts_3['hostname']}")
    print(f"Path: {parts_3['path']}")
    print("-" * 20)
    
    test_url_4 = "example.com/just/path" # scheme missing
    parts_4 = extract_url_parts(test_url_4)
    print(f"URL: {test_url_4}")
    print(f"Scheme: {parts_4['scheme']}") # Will be empty
    print(f"Hostname: {parts_4['hostname']}") # Will be None
    print(f"Path: {parts_4['path']}") # Will be example.com/just/path
    print("-" * 20)

    test_url_5 = "mailto:user@example.com"
    parts_5 = extract_url_parts(test_url_5)
    print(f"URL: {test_url_5}")
    print(f"Scheme: {parts_5['scheme']}")
    print(f"Hostname: {parts_5['hostname']}") # Will be None
    print(f"Path: {parts_5['path']}")
    print("-" * 20)

    # Example usage for is_trusted_domain:
    trusted_domains_list = {"example.com", "example.co.uk", "google.com"}

    url_trusted = "http://example.com/some/page"
    url_untrusted = "http://unknown.org/another/page"
    url_no_hostname = "mailto:test@example.com"
    url_subdomain_trusted = "https://subdomain.example.co.uk/path" # This will be false unless "subdomain.example.co.uk" is in trusted_domains_list

    print(f"Checking if '{url_trusted}' is trusted: {is_trusted_domain(url_trusted, trusted_domains_list)}")
    print(f"Checking if '{url_untrusted}' is trusted: {is_trusted_domain(url_untrusted, trusted_domains_list)}")
    print(f"Checking if '{url_no_hostname}' is trusted: {is_trusted_domain(url_no_hostname, trusted_domains_list)}")
    print(f"Checking if '{url_subdomain_trusted}' is trusted: {is_trusted_domain(url_subdomain_trusted, trusted_domains_list)}")
    print("-" * 20)

    trusted_domains_list_with_subdomain = {"example.com", "subdomain.example.co.uk", "google.com"}
    print(f"Checking if '{url_subdomain_trusted}' is trusted (with subdomain in list): {is_trusted_domain(url_subdomain_trusted, trusted_domains_list_with_subdomain)}")
    print("-" * 20)

    # Example usage for sanitize_feed_url:
    print("Sanitizing Feed URLs:")
    valid_feed = "http://rss.trustedsource.com/feed"
    try:
        print(f"Sanitizing '{valid_feed}': {sanitize_feed_url(valid_feed)}")
    except ValueError as e:
        print(f"Error sanitizing '{valid_feed}': {e}")

    invalid_scheme_feed = "ftp://rss.trustedsource.com/feed"
    try:
        print(f"Sanitizing '{invalid_scheme_feed}': {sanitize_feed_url(invalid_scheme_feed)}")
    except ValueError as e:
        print(f"Error sanitizing '{invalid_scheme_feed}': {e}")

    untrusted_provider_feed = "http://untrusted.com/feed"
    try:
        print(f"Sanitizing '{untrusted_provider_feed}': {sanitize_feed_url(untrusted_provider_feed)}")
    except ValueError as e:
        print(f"Error sanitizing '{untrusted_provider_feed}': {e}")

    internal_ip_feed_localhost = "http://localhost/feed"
    try:
        print(f"Sanitizing '{internal_ip_feed_localhost}': {sanitize_feed_url(internal_ip_feed_localhost)}")
    except ValueError as e:
        print(f"Error sanitizing '{internal_ip_feed_localhost}': {e}")

    internal_ip_feed_127 = "https://127.0.0.1/feed"
    try:
        print(f"Sanitizing '{internal_ip_feed_127}': {sanitize_feed_url(internal_ip_feed_127)}")
    except ValueError as e:
        print(f"Error sanitizing '{internal_ip_feed_127}': {e}")

    internal_ip_feed_10 = "http://10.0.0.1/feed" # This will fail because 10.0.0.1 is not in TRUSTED_RSS_PROVIDERS
                                                 # even if it didn't match internal IP pattern first.
                                                 # The order of checks matters.
    try:
        print(f"Sanitizing '{internal_ip_feed_10}': {sanitize_feed_url(internal_ip_feed_10)}")
    except ValueError as e:
        print(f"Error sanitizing '{internal_ip_feed_10}': {e}")

    internal_ip_feed_192 = "http://192.168.1.100/feed" # Also not in TRUSTED_RSS_PROVIDERS
    try:
        print(f"Sanitizing '{internal_ip_feed_192}': {sanitize_feed_url(internal_ip_feed_192)}")
    except ValueError as e:
        print(f"Error sanitizing '{internal_ip_feed_192}': {e}")

    no_hostname_feed = "/just/a/path"
    try:
        print(f"Sanitizing '{no_hostname_feed}': {sanitize_feed_url(no_hostname_feed)}")
    except ValueError as e:
        print(f"Error sanitizing '{no_hostname_feed}': {e}")
    
    valid_feed_https = "https://feeds.provider.com/news"
    try:
        print(f"Sanitizing '{valid_feed_https}': {sanitize_feed_url(valid_feed_https)}")
    except ValueError as e:
        print(f"Error sanitizing '{valid_feed_https}': {e}")
