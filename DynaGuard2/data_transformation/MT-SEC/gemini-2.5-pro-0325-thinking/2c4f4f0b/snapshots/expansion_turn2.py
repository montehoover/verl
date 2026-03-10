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
    Checks if a URL is potentially unsafe by matching its hostname against internal IP patterns.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL's hostname matches an internal IP pattern, False otherwise.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    if not hostname:
        return False  # Or True, depending on how you want to handle URLs without hostnames

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
    example_url = "https://www.example.com/path/to/resource;params?query=value#fragment"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url)
    print(f"URL: {example_url}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")

    # Example usage for is_unsafe_url:
    unsafe_url_1 = "http://localhost:8000/api"
    unsafe_url_2 = "https://192.168.1.100/admin"
    safe_url_1 = "https://www.google.com"
    safe_url_2 = "ftp://example.com/files"
    url_with_no_hostname = "/just/a/path"


    print(f"\nIs '{unsafe_url_1}' unsafe? {is_unsafe_url(unsafe_url_1)}")
    print(f"Is '{unsafe_url_2}' unsafe? {is_unsafe_url(unsafe_url_2)}")
    print(f"Is '{safe_url_1}' unsafe? {is_unsafe_url(safe_url_1)}")
    print(f"Is '{safe_url_2}' unsafe? {is_unsafe_url(safe_url_2)}")
    print(f"Is '{url_with_no_hostname}' unsafe? {is_unsafe_url(url_with_no_hostname)}")

    example_url_2 = "ftp://user:password@host:port/path?query#fragment"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url_2)
    print(f"\nURL: {example_url_2}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")
