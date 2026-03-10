import re
from urllib.parse import urlparse

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def extract_url_components(url: str) -> tuple[str, str, str, str, str, str]:
    """
    Breaks down a URL into its components.

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

def is_internal_network(url: str) -> bool:
    """
    Checks if the URL points to an internal network.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL's hostname matches an internal IP pattern, False otherwise.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname  # Use .hostname to get just the host, without port

    if not hostname:
        return False

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False

if __name__ == '__main__':
    # Example usage for extract_url_components:
    example_url = "http://user:pass@www.example.com:80/path/to/resource;params?query=value#fragment"
    scheme, netloc, path, params, query, fragment = extract_url_components(example_url)
    print(f"URL: {example_url}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")

    # Example usage for is_internal_network:
    internal_urls = [
        "http://localhost/admin",
        "http://127.0.0.1/dashboard",
        "http://10.0.0.5/internal_site",
        "http://192.168.1.100/config",
        "https://192.168.0.1:8080/path"
    ]
    external_urls = [
        "http://www.example.com",
        "https://google.com",
        "ftp://ftp.example.org/files"
    ]

    print("\nChecking internal URLs:")
    for url_to_check in internal_urls:
        print(f"  URL: {url_to_check}, Is internal: {is_internal_network(url_to_check)}")

    print("\nChecking external URLs:")
    for url_to_check in external_urls:
        print(f"  URL: {url_to_check}, Is internal: {is_internal_network(url_to_check)}")

    example_url_2 = "ftp://ftp.example.com/files/archive.zip"
    scheme, netloc, path, params, query, fragment = extract_url_components(example_url_2)
    print(f"\nURL: {example_url_2}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")
