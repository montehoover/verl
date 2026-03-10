from urllib.parse import urlparse

def check_safe_domain(url_string, safe_domains):
    """
    Extracts the hostname from a URL string and checks if it's in a list/set of safe domains.

    Args:
        url_string: The URL string to parse.
        safe_domains: A list or set of known safe hostnames.

    Returns:
        "safe" if the hostname is found and is in safe_domains, "unsafe" otherwise.
    """
    hostname = None
    try:
        parsed_url = urlparse(url_string)
        # Use hostname to check against safe_domains, ignoring port numbers
        hostname = parsed_url.hostname
    except Exception:
        # Includes ValueError for invalid URLs like "invalid-url-string"
        # Treat parsing errors or inability to extract hostname as unsafe
        return "unsafe"

    if hostname and hostname in safe_domains:
        return "safe"
    else:
        # Hostname not found (e.g., for "example.com" without scheme, hostname is None),
        # or hostname is empty, or not in the safe list
        return "unsafe"

if __name__ == '__main__':
    # Define a set of safe hostnames (using a set for efficient lookup)
    SAFE_DOMAINS_SET = {
        "www.example.com",
        "subdomain.example.co.uk",
        "example.org",
        "localhost"  # For URLs like http://localhost:3000, hostname is 'localhost'
    }

    # Example Usage
    urls_to_test = [
        "http://www.example.com/path/to/page?query=string#fragment",      # Expected: safe
        "https://subdomain.example.co.uk:8080/another/path",             # Expected: safe (port ignored)
        "ftp://user:password@example.org/resource",                      # Expected: safe
        "http://www.malicious.com/evil/script",                          # Expected: unsafe
        "example.com",  # No scheme, urlparse treats as path, hostname is None. Expected: unsafe
        "invalid-url-string",                                            # Invalid URL. Expected: unsafe
        "http://localhost:3000/test",                                    # Expected: safe (localhost is safe)
        "//another-safe-domain.com/path"                                 # Hostname 'another-safe-domain.com'. Expected: unsafe
    ]

    print("Checking URLs against safe domains (hostname based check):")
    for url in urls_to_test:
        # For display purposes, let's show what hostname is extracted
        extracted_hostname_display = "N/A"
        try:
            parsed_url_display = urlparse(url)
            if parsed_url_display.hostname:
                extracted_hostname_display = parsed_url_display.hostname
            elif parsed_url_display.netloc: # e.g. if scheme missing, netloc might be domain, hostname None
                extracted_hostname_display = f"'{parsed_url_display.netloc}' (netloc, but hostname is None)"
            else: # e.g. "example.com" (no scheme) -> path, hostname is None
                 extracted_hostname_display = "None (likely treated as path)"
        except Exception:
            extracted_hostname_display = "Error parsing URL"

        status = check_safe_domain(url, SAFE_DOMAINS_SET)
        print(f"URL: {url:<65} Extracted Hostname for check: {extracted_hostname_display:<45} Status: {status}")

    print("\n--- Further specific test cases ---")

    # Test case: URL with a port, domain is in safe list
    url_with_port_safe = "http://www.example.com:1234/secure"
    status_wp_safe = check_safe_domain(url_with_port_safe, SAFE_DOMAINS_SET)
    print(f"URL: {url_with_port_safe:<65} Hostname: {urlparse(url_with_port_safe).hostname:<45} Status: {status_wp_safe}") # safe

    # Test case: URL whose hostname is not in the safe list
    url_unsafe_domain = "http://unknown.domain.net/page"
    status_unsafe = check_safe_domain(url_unsafe_domain, SAFE_DOMAINS_SET)
    print(f"URL: {url_unsafe_domain:<65} Hostname: {urlparse(url_unsafe_domain).hostname:<45} Status: {status_unsafe}") # unsafe

    # Test case: URL that urlparse treats as a path because scheme is missing
    # urlparse("myhost.com/resource").hostname is None
    url_no_scheme = "myhost.com/resource"
    status_no_scheme = check_safe_domain(url_no_scheme, SAFE_DOMAINS_SET)
    print(f"URL: {url_no_scheme:<65} Hostname: {urlparse(url_no_scheme).hostname!r:<45} Status: {status_no_scheme}") # unsafe

    # Test case: URL with "//" prefix, ensuring hostname is parsed and checked
    # (assuming "example.com" itself is not in SAFE_DOMAINS_SET initially for this test)
    url_double_slash_unsafe = "//example.com/resource" # hostname is 'example.com'
    status_ds_unsafe = check_safe_domain(url_double_slash_unsafe, SAFE_DOMAINS_SET)
    print(f"URL: {url_double_slash_unsafe:<65} Hostname: {urlparse(url_double_slash_unsafe).hostname:<45} Status: {status_ds_unsafe}") # unsafe

    # Test with "example.com" added to a temporary safe list
    temp_safe_domains = SAFE_DOMAINS_SET.union({"example.com"})
    status_ds_safe = check_safe_domain(url_double_slash_unsafe, temp_safe_domains)
    print(f"URL: {url_double_slash_unsafe:<65} Hostname: {urlparse(url_double_slash_unsafe).hostname:<45} Status (with 'example.com' safe): {status_ds_safe}") # safe
