import re
from urllib.parse import urlparse

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def is_unsafe_url(url: str) -> bool:
    """
    Checks if a URL points to an internal or potentially unsafe IP address.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL is considered unsafe, False otherwise.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname  # Use .hostname to get just the host, without port

    if not hostname:
        return False  # Cannot determine hostname, assume not unsafe for this check

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False

def extract_url_parts(url: str) -> tuple[str, str, str, str, str, str]:
    """
    Extracts the components of a URL.

    Args:
        url: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, path, params, query, and fragment.
    """
    parsed_url = urlparse(url)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )

if __name__ == '__main__':
    # Example usage:
    example_url = "http://user:password@www.example.com:80/path/to/resource;params?query=value#fragment"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url)
    print(f"URL: {example_url}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")
    print(f"  Is Unsafe: {is_unsafe_url(example_url)}")

    example_url_2 = "https://docs.python.org/3/library/urllib.parse.html#urllib.parse.urlparse"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url_2)
    print(f"\nURL: {example_url_2}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")
    print(f"  Is Unsafe: {is_unsafe_url(example_url_2)}")

    # Test cases for is_unsafe_url
    unsafe_urls_to_test = [
        "http://localhost/admin",
        "http://127.0.0.1/data",
        "ftp://10.0.0.5/files",
        "http://192.168.1.100:8080/config",
        "https://192.168.0.1",
    ]
    safe_urls_to_test = [
        "https://www.google.com",
        "http://example.com",
        "ftp://public.ftp-servers.net",
    ]

    print("\nTesting unsafe URLs:")
    for u_url in unsafe_urls_to_test:
        print(f"  URL: {u_url}, Is Unsafe: {is_unsafe_url(u_url)}")

    print("\nTesting safe URLs:")
    for s_url in safe_urls_to_test:
        print(f"  URL: {s_url}, Is Unsafe: {is_unsafe_url(s_url)}")

    example_url_2 = "https://docs.python.org/3/library/urllib.parse.html#urllib.parse.urlparse"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url_2)
    print(f"\nURL: {example_url_2}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")
