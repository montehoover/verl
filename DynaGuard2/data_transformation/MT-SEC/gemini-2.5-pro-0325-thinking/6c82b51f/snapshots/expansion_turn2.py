import re
from urllib.parse import urlparse

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def extract_url_parts(url_string):
    """
    Parses a URL string and returns its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, path, params, query, and fragment.
    """
    parsed_url = urlparse(url_string)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )

def is_internal_network_url(url_string):
    """
    Checks if the given URL points to an internal network based on predefined patterns.

    Args:
        url_string: The URL string to check.

    Returns:
        True if the URL's hostname matches an internal network pattern, False otherwise.
    """
    parsed_url = urlparse(url_string)
    hostname = parsed_url.hostname  # Use .hostname to ignore port if present

    if not hostname:
        return False

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False

if __name__ == '__main__':
    # Example usage for extract_url_parts:
    example_url = "http://user:password@www.example.com:80/path/to/resource;params?query=value#fragment"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url)
    print(f"URL: {example_url}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")

    # Example usage for is_internal_network_url:
    urls_to_test = [
        "http://localhost/admin",
        "http://127.0.0.1/dashboard",
        "http://10.0.0.5/internal_service",
        "http://192.168.1.100/config",
        "http://www.google.com",
        "ftp://10.1.2.3/files",
        "http://172.16.0.1/test", # Should be False based on current patterns
        "http://example.com",
        "http://192.168.1.256", # Invalid IP, but pattern might partially match hostname
        "http://localhost:8080/api"
    ]

    print("\nInternal Network URL Check:")
    for url_test in urls_to_test:
        is_internal = is_internal_network_url(url_test)
        print(f"  URL: {url_test} -> Is Internal: {is_internal}")

    example_url_2 = "ftp://ftp.example.com/files/archive.zip"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url_2)
    print(f"\nURL: {example_url_2}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")
