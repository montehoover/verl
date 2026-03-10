import ipaddress
from urllib.parse import urlparse

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the scheme, hostname, and path of the URL.
        Returns None if the URL cannot be parsed.
    """
    try:
        parsed_url = urlparse(url_string)
        return {
            "scheme": parsed_url.scheme,
            "hostname": parsed_url.hostname,
            "path": parsed_url.path,
            "params": parsed_url.params,
            "query": parsed_url.query,
            "fragment": parsed_url.fragment,
            "port": parsed_url.port,
        }
    except Exception: # pylint: disable=broad-except
        # urlparse can raise various exceptions for malformed URLs,
        # though it's generally robust. Returning None for simplicity.
        return None

def is_valid_external_url(url_string):
    """
    Validates if a URL is external and uses http/https.

    Checks for:
    - Scheme is 'http' or 'https'.
    - Hostname is not 'localhost'.
    - Hostname is not a private, loopback, or link-local IP address.

    Args:
        url_string: The URL string to validate.

    Returns:
        True if the URL is valid for external use, False otherwise.
    """
    parts = extract_url_parts(url_string)

    if not parts:
        return False

    scheme = parts.get("scheme")
    hostname = parts.get("hostname")

    if not scheme or not hostname:
        return False

    if scheme.lower() not in ("http", "https"):
        return False

    hostname_lower = hostname.lower()
    if hostname_lower == "localhost":
        return False

    try:
        ip_addr = ipaddress.ip_address(hostname_lower)
        if ip_addr.is_private or ip_addr.is_loopback or ip_addr.is_link_local:
            return False
    except ValueError:
        # Not an IP address, so it's a domain name.
        # This is generally acceptable for an "external" URL unless it's 'localhost' (already checked).
        pass
    
    return True

if __name__ == '__main__':
    test_url_1 = "http://www.example.com/path/to/resource?name=test#fragment"
    parts_1 = extract_url_parts(test_url_1)
    print(f"Parts for {test_url_1}: {parts_1}")

    test_url_2 = "https://subdomain.example.co.uk:8080/another/path.html?key1=value1&key2=value2"
    parts_2 = extract_url_parts(test_url_2)
    print(f"Parts for {test_url_2}: {parts_2}")

    test_url_3 = "ftp://user:password@ftp.example.com/files/archive.zip"
    parts_3 = extract_url_parts(test_url_3)
    print(f"Parts for {test_url_3}: {parts_3}")

    # Example of a URL without a path
    test_url_4 = "mailto:user@example.com"
    parts_4 = extract_url_parts(test_url_4)
    print(f"Parts for {test_url_4}: {parts_4}")

    # Example of a relative URL (hostname will be None)
    test_url_5 = "/relative/path?query=yes"
    parts_5 = extract_url_parts(test_url_5)
    print(f"Parts for {test_url_5}: {parts_5}")
    
    # Example of an invalid URL (might not raise an error but return empty parts)
    test_url_6 = "this is not a url"
    parts_6 = extract_url_parts(test_url_6)
    print(f"Parts for {test_url_6}: {parts_6}")

    print("\n--- URL Validation Tests ---")
    urls_to_validate = [
        ("http://www.example.com", True),
        ("https://www.google.com", True),
        ("http://localhost/path", False),
        ("https://127.0.0.1/secure", False),
        ("http://10.0.0.1/internal", False),
        ("https://192.168.1.100/admin", False),
        ("http://172.16.0.5/config", False), # Private IP
        ("http://169.254.1.1/local", False), # Link-local IP
        ("ftp://ftp.example.com/file.txt", False), # Invalid scheme
        ("http://example.com", True),
        ("https://nonexistentdomain12345.com", True), # Domain itself is fine, resolution not checked
        ("http://1.2.3.4", True), # Public IP
        ("/relative/path", False), # No scheme/hostname
        ("mailto:test@example.com", False), # No scheme/hostname for http/https
        ("http://", False), # No hostname
        ("this is not a url", False),
    ]

    for url, expected_validity in urls_to_validate:
        is_valid = is_valid_external_url(url)
        print(f"URL: '{url}', Valid: {is_valid}, Expected: {expected_validity}, Pass: {is_valid == expected_validity}")
