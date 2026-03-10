from urllib.parse import urlparse
import ipaddress
import socket

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the URL components (scheme, hostname, path, params, query, fragment).
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "hostname": parsed_url.hostname,
        "path": parsed_url.path,
        "params": parsed_url.params,
        "query": parsed_url.query,
        "fragment": parsed_url.fragment,
        "netloc": parsed_url.netloc, # often includes port
        "port": parsed_url.port
    }

def check_url_security(url_string):
    """
    Checks if a URL is secure based on its scheme and network address.

    Args:
        url_string: The URL string to check.

    Returns:
        "secure" if the URL uses 'https' and does not point to an internal
        or loopback IP address, "insecure" otherwise.
    """
    try:
        parsed_url = urlparse(url_string)

        if parsed_url.scheme.lower() != 'https':
            return "insecure"

        hostname = parsed_url.hostname
        if not hostname:
            return "insecure" # URLs like file:/// or mailto:

        ip_addr_obj = None
        try:
            # Try to interpret hostname as an IP address directly
            ip_addr_obj = ipaddress.ip_address(hostname)
        except ValueError:
            # If not an IP, assume it's a domain name and resolve it
            try:
                ip_str = socket.gethostbyname(hostname)
                ip_addr_obj = ipaddress.ip_address(ip_str)
            except (socket.gaierror, OSError): # Covers DNS resolution failures and other network errors
                return "insecure" # Cannot resolve, treat as insecure

        if ip_addr_obj:
            if ip_addr_obj.is_private or ip_addr_obj.is_loopback:
                return "insecure" # Points to an internal or loopback address
        else:
            # This case should ideally not be reached if hostname was present
            # and resolution was attempted.
            return "insecure"


        return "secure"
    except Exception:
        # Catch any other parsing errors or unexpected issues
        return "insecure"

if __name__ == '__main__':
    # Example usage for extract_url_parts:
    test_url = "https://www.example.com:8080/path/to/resource?param1=value1&param2=value2#section1"
    url_parts = extract_url_parts(test_url)
    print(f"URL: {test_url}")
    for key, value in url_parts.items():
        print(f"{key}: {value}")

    print("\nAnother example:")
    test_url_2 = "ftp://user:password@ftp.example.co.uk/files/archive.zip;type=i"
    url_parts_2 = extract_url_parts(test_url_2)
    print(f"URL: {test_url_2}")
    for key, value in url_parts_2.items():
        print(f"{key}: {value}")

    print("\n--- URL Security Checks ---")
    urls_to_check = [
        "https://www.google.com",
        "http://example.com",
        "https://localhost",
        "https://127.0.0.1",
        "https://192.168.1.100",
        "https://10.0.0.5",
        "ftp://internal.service.local",
        "https://my-public-service.example.com",
        "https://nonexistentdomain12345.com", # Test DNS failure
        "file:///etc/passwd"
    ]

    for url in urls_to_check:
        security_status = check_url_security(url)
        print(f"URL: {url} -> Status: {security_status}")
