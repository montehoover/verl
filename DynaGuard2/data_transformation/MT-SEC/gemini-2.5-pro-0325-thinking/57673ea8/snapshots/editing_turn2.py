import socket
import ipaddress
from urllib.parse import urlparse

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

if __name__ == '__main__':
    # Example usage for extract_url_parts (existing):
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
        # Add a known public IPv6 address if possible for a True case,
        # but this depends on network setup and DNS.
        # For now, we rely on named hosts that resolve to public IPs.
        # Example: "https://[2001:db8::1]/publicipv6" (if resolvable and public)
    ]

    for url, expected_secure in urls_to_test_security:
        is_secure = check_url_security(url)
        status = "Pass" if is_secure == expected_secure else "Fail"
        print(f"URL: {url}, Expected Secure: {expected_secure}, Actual Secure: {is_secure} -> {status}")
