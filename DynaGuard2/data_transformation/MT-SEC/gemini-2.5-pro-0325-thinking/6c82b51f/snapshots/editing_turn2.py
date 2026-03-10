from urllib.parse import urlparse
import ipaddress

def is_url_secure(url_string):
    """
    Checks if a URL is secure according to specified criteria:
    - Scheme must be 'http' or 'https'.
    - Does not point to 'localhost'.
    - Does not point to private IP ranges.

    Args:
        url_string: The URL string to check.

    Returns:
        True if the URL is considered secure, False otherwise.
    """
    try:
        parsed_url = urlparse(url_string)

        # Check 1: Scheme must be http or https
        if parsed_url.scheme.lower() not in ('http', 'https'):
            return False

        # Check 2: Hostname must exist for http/https URLs
        hostname = parsed_url.hostname
        if not hostname:
            return False
        
        hostname = hostname.lower() # Normalize hostname for checks

        # Check 3: Hostname must not be 'localhost'
        if hostname == 'localhost':
            return False

        # Check 4: Hostname, if an IP address, must not be private or loopback
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback:
                return False
        except ValueError:
            # Hostname is not a valid IP address (e.g., it's a domain name).
            # This is acceptable for non-internal addresses.
            pass
        
        return True # All checks passed

    except Exception:
        # Catch any other parsing errors or unexpected issues with the URL string
        return False

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components like scheme, netloc, and path.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the 'scheme', 'netloc', and 'path' of the URL.
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "netloc": parsed_url.netloc,
        "path": parsed_url.path,
    }

if __name__ == '__main__':
    # Example usage:
    urls_to_test = [
        "http://www.example.com/path/to/resource?query=param#fragment",
        "https://docs.python.org/3/library/urllib.parse.html",
        "ftp://example.com/resource", # Insecure scheme
        "http://localhost/admin",     # Insecure host (localhost)
        "https://127.0.0.1/secure",   # Insecure host (loopback IP)
        "http://10.0.0.5/internal",   # Insecure host (private IP)
        "https://192.168.1.100/lan",  # Insecure host (private IP)
        "http://172.16.0.1/private",  # Insecure host (private IP)
        "https://example.com",
        "http:///path/only",          # Invalid http URL (no hostname)
        "malformed-url"               # Malformed URL
    ]

    for url in urls_to_test:
        secure_status = is_url_secure(url)
        # Original parts extraction for context if needed (optional)
        # parts = extract_url_parts(url) 
        # print(f"URL: {url}, Scheme: {parts['scheme']}, Netloc: {parts['netloc']}, Path: {parts['path']}, Secure: {secure_status}")
        print(f"URL: {url}, Secure: {secure_status}")

    # Example of using extract_url_parts if still needed separately
    print("\nExample of extract_url_parts:")
    sample_url_for_parts = "https://www.another-example.com/test"
    parts = extract_url_parts(sample_url_for_parts)
    print(f"URL: {sample_url_for_parts}")
    print(f"Scheme: {parts['scheme']}")
    print(f"Netloc: {parts['netloc']}")
    print(f"Path: {parts['path']}")
