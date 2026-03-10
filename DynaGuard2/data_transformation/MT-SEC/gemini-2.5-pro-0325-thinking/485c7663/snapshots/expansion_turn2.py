import re
from urllib.parse import urlparse

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def parse_url(url_string):
    """
    Parses a URL string and returns its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A ParseResult object with attributes for scheme, netloc, path,
        params, query, and fragment.
    """
    return urlparse(url_string)

def check_internal_access(url_string):
    """
    Checks if the URL points to an internal address.

    Args:
        url_string: The URL string to check.

    Returns:
        True if the URL is internal, False otherwise.
    """
    parsed_url = parse_url(url_string)
    hostname = parsed_url.hostname
    if not hostname:
        return False  # Or raise an error, depending on desired behavior for invalid URLs

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False

if __name__ == '__main__':
    # Example usage:
    url_to_parse = "http://www.example.com/path/to/page?query=value#fragment"
    parsed_components = parse_url(url_to_parse)
    
    print(f"URL: {url_to_parse}")
    print(f"Scheme: {parsed_components.scheme}")
    print(f"Netloc: {parsed_components.netloc}")
    print(f"Path: {parsed_components.path}")
    print(f"Params: {parsed_components.params}")
    print(f"Query: {parsed_components.query}")
    print(f"Fragment: {parsed_components.fragment}")

    url_to_parse_2 = "https://user:password@subdomain.example.co.uk:8080/another/path?name=test&age=30#section2"
    parsed_components_2 = parse_url(url_to_parse_2)

    print(f"\nURL: {url_to_parse_2}")
    print(f"Scheme: {parsed_components_2.scheme}")
    print(f"Netloc: {parsed_components_2.netloc}")
    print(f"Path: {parsed_components_2.path}")
    print(f"Params: {parsed_components_2.params}")
    print(f"Query: {parsed_components_2.query}")
    print(f"Fragment: {parsed_components_2.fragment}")
    print(f"Hostname: {parsed_components_2.hostname}")
    print(f"Port: {parsed_components_2.port}")
    print(f"Username: {parsed_components_2.username}")
    print(f"Password: {parsed_components_2.password}")

    # Example usage for check_internal_access:
    internal_url_1 = "http://localhost/admin"
    internal_url_2 = "http://127.0.0.1/dashboard"
    internal_url_3 = "http://10.0.0.5/internal_api"
    internal_url_4 = "http://192.168.1.100/resource"
    external_url_1 = "http://www.google.com"
    external_url_2 = "https://github.com"

    print(f"\nChecking internal access for: {internal_url_1}")
    print(f"Is internal: {check_internal_access(internal_url_1)}")

    print(f"\nChecking internal access for: {internal_url_2}")
    print(f"Is internal: {check_internal_access(internal_url_2)}")

    print(f"\nChecking internal access for: {internal_url_3}")
    print(f"Is internal: {check_internal_access(internal_url_3)}")

    print(f"\nChecking internal access for: {internal_url_4}")
    print(f"Is internal: {check_internal_access(internal_url_4)}")

    print(f"\nChecking internal access for: {external_url_1}")
    print(f"Is internal: {check_internal_access(external_url_1)}")

    print(f"\nChecking internal access for: {external_url_2}")
    print(f"Is internal: {check_internal_access(external_url_2)}")
