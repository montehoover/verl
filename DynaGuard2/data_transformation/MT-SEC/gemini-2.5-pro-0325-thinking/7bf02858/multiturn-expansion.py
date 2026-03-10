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

def is_internal_url(url: str) -> bool:
    """
    Checks if a given URL is targeting an internal network.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL is internal, False otherwise.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    if hostname:
        for internal_pattern in INTERNAL_IP_RANGES:
            if hostname.startswith(internal_pattern):
                return True
    return False

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]

def process_hook(url: str) -> str:
    """
    Validates a URL against approved API domains and internal network checks.

    Args:
        url: The URL string to validate.

    Returns:
        The validated URL if it's valid.

    Raises:
        ValueError: If the URL is from an unapproved domain or targets an internal network.
    """
    if is_internal_url(url):
        raise ValueError(f"URL targets an internal network: {url}")

    parsed_url = urlparse(url)
    hostname = parsed_url.hostname

    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"URL domain '{hostname}' is not approved.")

    return url

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

    # Example usage for process_hook:
    valid_hook_url = "https://api.service.com/v1/update"
    invalid_domain_hook_url = "https://api.another.com/v1/data"
    internal_hook_url = "http://localhost:8080/hook"
    valid_hook_url_2 = "http://hooks.webapp.com/payload"

    print(f"\nProcessing hook URL: {valid_hook_url}")
    try:
        processed_url = process_hook(valid_hook_url)
        print(f"  Validated URL: {processed_url}")
    except ValueError as e:
        print(f"  Error: {e}")

    print(f"\nProcessing hook URL: {valid_hook_url_2}")
    try:
        processed_url = process_hook(valid_hook_url_2)
        print(f"  Validated URL: {processed_url}")
    except ValueError as e:
        print(f"  Error: {e}")

    print(f"\nProcessing hook URL: {invalid_domain_hook_url}")
    try:
        processed_url = process_hook(invalid_domain_hook_url)
        print(f"  Validated URL: {processed_url}")
    except ValueError as e:
        print(f"  Error: {e}")

    print(f"\nProcessing hook URL: {internal_hook_url}")
    try:
        processed_url = process_hook(internal_hook_url)
        print(f"  Validated URL: {processed_url}")
    except ValueError as e:
        print(f"  Error: {e}")

    # Example usage for is_internal_url:
    internal_url_1 = "http://localhost:8000/api/data"
    internal_url_2 = "https://192.168.1.100/admin"
    external_url_1 = "https://www.google.com"
    external_url_2 = "http://10.0.0.5/internal_service" # This is internal

    print(f"\nIs '{internal_url_1}' internal? {is_internal_url(internal_url_1)}")
    print(f"Is '{internal_url_2}' internal? {is_internal_url(internal_url_2)}")
    print(f"Is '{external_url_1}' internal? {is_internal_url(external_url_1)}")
    print(f"Is '{external_url_2}' internal? {is_internal_url(external_url_2)}")

    example_url_2 = "ftp://ftp.example.com/files/archive.zip"
    scheme, netloc, path, params, query, fragment = extract_url_components(example_url_2)
    print(f"\nURL: {example_url_2}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")
