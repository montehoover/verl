import ipaddress
from urllib.parse import urlparse

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components (scheme, netloc, path).

    Args:
        url_string (str): The URL string to parse.

    Returns:
        dict: A dictionary containing the scheme, netloc, and path of the URL.
              Returns None for parts that are not present.
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme if parsed_url.scheme else None,
        "netloc": parsed_url.netloc if parsed_url.netloc else None,
        "path": parsed_url.path if parsed_url.path else None,
    }

def check_url_security(url_string):
    """
    Checks if a URL uses a secure scheme ('https') and does not point to an internal network address.

    Args:
        url_string (str): The URL string to check.

    Returns:
        bool: True if the URL has 'https' scheme and is not an internal/loopback/link-local 
              address, False otherwise.
    """
    try:
        # urlparse is robust for string inputs, but other types could cause errors.
        parsed_url = urlparse(str(url_string)) 
    except Exception: # Catching generic Exception for unforeseen parsing issues with non-string inputs
        return False

    # 1. Check for secure scheme ('https')
    # Scheme must be present and be 'https'
    if not parsed_url.scheme or parsed_url.scheme.lower() != 'https':
        return False

    hostname = parsed_url.hostname
    # 2. Check if hostname is present
    # For a valid 'https' URL, a hostname is expected.
    # e.g., urlparse('https://') results in hostname=None.
    if not hostname:
        return False

    # 3. Check for 'localhost'
    if hostname.lower() == 'localhost':
        return False # Internal

    # 4. Check if hostname is an IP address and if it's internal/loopback/link-local/unspecified
    try:
        ip_obj = ipaddress.ip_address(hostname)
        if ip_obj.is_private or \
           ip_obj.is_loopback or \
           ip_obj.is_link_local or \
           ip_obj.is_unspecified: # e.g., 0.0.0.0 or ::
            return False # Internal or non-routable IP
    except ValueError:
        # Hostname is not a valid IP address, so it's treated as a domain name.
        # For this function, non-IP, non-localhost domain names are considered potentially external.
        # DNS resolution to check if the domain resolves to an internal IP is out of scope.
        pass # Continue, as it might be an external domain name

    return True # Passes all checks: 'https' scheme and not an identified internal/special address

if __name__ == '__main__':
    # Example usage:
    test_url_1 = "http://www.example.com/path/to/resource?query=param#fragment"
    parts_1 = extract_url_parts(test_url_1)
    print(f"URL: {test_url_1}")
    print(f"Scheme: {parts_1['scheme']}")
    print(f"Netloc: {parts_1['netloc']}")
    print(f"Path: {parts_1['path']}")
    print("-" * 20)

    test_url_2 = "ftp://ftp.example.org/files/"
    parts_2 = extract_url_parts(test_url_2)
    print(f"URL: {test_url_2}")
    print(f"Scheme: {parts_2['scheme']}")
    print(f"Netloc: {parts_2['netloc']}")
    print(f"Path: {parts_2['path']}")
    print("-" * 20)

    test_url_3 = "/just/a/path" # Relative URL
    parts_3 = extract_url_parts(test_url_3)
    print(f"URL: {test_url_3}")
    print(f"Scheme: {parts_3['scheme']}") # Expected: None
    print(f"Netloc: {parts_3['netloc']}") # Expected: None
    print(f"Path: {parts_3['path']}")   # Expected: /just/a/path
    print("-" * 20)

    test_url_4 = "www.example.com" # URL without scheme
    parts_4 = extract_url_parts(test_url_4)
    print(f"URL: {test_url_4}")
    print(f"Scheme: {parts_4['scheme']}") # Expected: None
    print(f"Netloc: {parts_4['netloc']}") # Expected: None (or www.example.com depending on strictness, urlparse treats it as path)
    print(f"Path: {parts_4['path']}")   # Expected: www.example.com
    print("-" * 20)

    print("\nTesting check_url_security:")
    urls_to_test = {
        "https://www.google.com": True,
        "http://www.google.com": False,    # Scheme is not 'https'
        "https://localhost/somepage": False, # Hostname is 'localhost'
        "https://127.0.0.1/secure": False,   # Loopback IP address
        "https://192.168.1.100/admin": False, # Private IPv4 address
        "https://10.0.0.1": False,           # Private IPv4 address
        "https://172.16.0.1": False,         # Private IPv4 address
        "ftp://example.com/file": False,   # Scheme is not 'https'
        "https://example.com": True,
        "https://169.254.1.1": False,       # Link-local IPv4 address
        "https://[::1]/test": False,         # Loopback IPv6 address
        "https://[fc00::1]/test": False,     # Unique Local IPv6 Address (private)
        "https://[fe80::1]/test": False,     # Link-local IPv6 address
        "https://8.8.8.8": True,             # Public IP address
        "not_a_url_string": False,         # Parsed as path, no scheme
        "https://": False,                   # Scheme 'https', but no hostname
        "//example.com/path": False,       # Scheme-relative, parsed_url.scheme is empty
        "https://www.example.com/path?query=value#fragment": True,
        "https://0.0.0.0": False,            # Unspecified IP address
        "https://[::]": False,               # Unspecified IPv6 address
        123: False,                        # Non-string input
    }

    for url, expected in urls_to_test.items():
        result = check_url_security(url)
        status = "PASS" if result == expected else "FAIL"
        print(f"URL: \"{url}\", Secure & External: {result} (Expected: {expected}) - {status}")
