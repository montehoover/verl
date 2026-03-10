import re
from urllib.parse import urlparse

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def is_internal_network_url(url: str) -> bool:
    """
    Checks if the URL points to an internal network address.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL's hostname matches an internal IP pattern, False otherwise.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    if not hostname:
        return False  # Or handle as an error, depending on requirements

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False

def extract_url_parts(url: str):
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
    example_url = "http://user:pass@www.example.com:80/path/to/resource;params?query=value#fragment"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url)
    print(f"URL: {example_url}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")

    # Example usage for is_internal_network_url:
    internal_url_1 = "http://localhost:8000/api"
    internal_url_2 = "http://192.168.1.100/admin"
    external_url_1 = "https://www.google.com"
    no_hostname_url = "/just/a/path" # urlparse will result in no hostname

    print(f"\nURL: {internal_url_1}, Is internal? {is_internal_network_url(internal_url_1)}")
    print(f"URL: {internal_url_2}, Is internal? {is_internal_network_url(internal_url_2)}")
    print(f"URL: {external_url_1}, Is internal? {is_internal_network_url(external_url_1)}")
    print(f"URL: {no_hostname_url}, Is internal? {is_internal_network_url(no_hostname_url)}")

    example_url_2 = "ftp://ftp.example.com/files/archive.zip"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url_2)
    print(f"\nURL: {example_url_2}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")
