from urllib.parse import urlparse

def extract_url_components(url: str) -> tuple[str, str, str, str, str, str]:
    """
    Extracts components from a given URL.

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

INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def is_internal_network(url: str) -> bool:
    """
    Checks if the URL belongs to an internal network.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL is internal, False otherwise.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.netloc.split(':')[0] # Remove port if present
    if not hostname:
        return False # Or raise an error, depending on desired behavior for invalid URLs

    for internal_range in INTERNAL_IP_RANGES:
        if hostname.startswith(internal_range):
            return True
    return False

if __name__ == '__main__':
    # Example usage for extract_url_components:
    example_url = "http://user:password@www.example.com:80/path/to/resource;params?query=value#fragment"
    components = extract_url_components(example_url)
    print(f"URL: {example_url}")
    print(f"Scheme: {components[0]}")
    print(f"Netloc: {components[1]}")
    print(f"Path: {components[2]}")
    print(f"Params: {components[3]}")
    print(f"Query: {components[4]}")
    print(f"Fragment: {components[5]}")

    example_url_2 = "https://docs.python.org/3/library/urllib.parse.html#urllib.parse.urlparse"
    components_2 = extract_url_components(example_url_2)
    print(f"\nURL: {example_url_2}")
    print(f"Scheme: {components_2[0]}")
    print(f"Netloc: {components_2[1]}")
    print(f"Path: {components_2[2]}")
    print(f"Params: {components_2[3]}")
    print(f"Query: {components_2[4]}")
    print(f"Fragment: {components_2[5]}")

    # Example usage for is_internal_network:
    internal_url_1 = "http://localhost:8000/api/data"
    internal_url_2 = "https://192.168.1.100/admin"
    internal_url_3 = "ftp://10.0.0.5/files"
    external_url_1 = "https://www.google.com"
    external_url_2 = "http://example.com"

    print(f"\nIs '{internal_url_1}' internal? {is_internal_network(internal_url_1)}")
    print(f"Is '{internal_url_2}' internal? {is_internal_network(internal_url_2)}")
    print(f"Is '{internal_url_3}' internal? {is_internal_network(internal_url_3)}")
    print(f"Is '{external_url_1}' internal? {is_internal_network(external_url_1)}")
    print(f"Is '{external_url_2}' internal? {is_internal_network(external_url_2)}")
