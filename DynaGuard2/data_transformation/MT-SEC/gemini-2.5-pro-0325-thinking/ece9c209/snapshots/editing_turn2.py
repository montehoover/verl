import ipaddress
from urllib.parse import urlparse

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components such as scheme, hostname, and path.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the scheme, hostname, and path of the URL.
        The values might be empty strings or None if a component is not present
        or if the URL is malformed in a way that urlparse handles gracefully.
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "hostname": parsed_url.hostname,
        "path": parsed_url.path,
    }

def check_url_security(url_string):
    """
    Verifies that the URL uses a secure scheme ('https') and does not point to
    any internal network addresses.

    Args:
        url_string: The URL string to check.

    Returns:
        True if the URL is considered secure, False otherwise.
    """
    parsed_url = urlparse(url_string)

    # Check for secure scheme
    if parsed_url.scheme != 'https':
        return False

    # Check for internal network addresses
    hostname = parsed_url.hostname
    if not hostname:
        # Cannot determine security without a hostname (e.g., relative paths, mailto)
        return False

    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_loopback or ip.is_private or ip.is_link_local:
            return False
    except ValueError:
        # Hostname is not a valid IP address, which is common for domain names.
        # For simplicity, we'll consider non-IP hostnames as potentially public.
        # A more robust solution might involve DNS resolution and checking against
        # known internal domain names, but that's beyond simple static analysis.
        # Common internal hostnames could also be checked here (e.g., 'localhost').
        if hostname == "localhost":
            return False
        pass # It's a domain name, proceed with scheme check (already done)

    return True

if __name__ == '__main__':
    # Example Usage for extract_url_parts
    print("--- URL Parts Extraction ---")
    urls_to_test_extraction = [
        "https://www.example.com/path/to/resource?query=param#fragment",
        "http://localhost:8080/api/v1/users",
        "ftp://ftp.example.com/files/archive.zip",
        "//example.com/path",  # Scheme-relative URL
        "/just/a/path",        # Only a path
        "mailto:user@example.com",
        "urn:isbn:0451450523"
    ]

    for url in urls_to_test_extraction:
        parts = extract_url_parts(url)
        print(f"URL: {url}")
        print(f"  Scheme: {parts['scheme']}")
        print(f"  Hostname: {parts['hostname']}")
        print(f"  Path: {parts['path']}")
        print("-" * 20)

    print("\n--- URL Security Check ---")
    urls_to_test_security = [
        "https://www.example.com/path/to/resource", # Secure
        "http://www.example.com/path/to/resource",  # Insecure (http)
        "https://localhost/api/v1/users",          # Insecure (localhost)
        "https://127.0.0.1/api/v1/users",          # Insecure (loopback IP)
        "https://192.168.1.100/dashboard",         # Insecure (private IP)
        "https://10.0.0.5/internal",               # Insecure (private IP)
        "https://172.16.0.1/data",                 # Insecure (private IP)
        "ftp://ftp.example.com/files/archive.zip", # Insecure (ftp)
        "//example.com/path",                      # Insecure (scheme-relative, implies http or https based on context, but urlparse gives no scheme)
        "/just/a/path",                            # Insecure (no scheme/host)
        "https://[::1]/test",                      # Insecure (loopback IPv6)
        "https://169.254.1.1/config"               # Insecure (link-local)
    ]

    for url in urls_to_test_security:
        is_secure = check_url_security(url)
        status = "Secure" if is_secure else "Insecure"
        print(f"URL: {url} -> Status: {status}")
        print("-" * 20)
