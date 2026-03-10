from urllib.parse import urlencode, urlparse

def build_url_with_query(base_url: str, path: str, query_params: dict = None) -> str:
    """
    Constructs a URL from a base URL, a path, and optional query parameters.

    Args:
        base_url: The base URL (e.g., "http://example.com").
                  Must be a valid URL.
        path: The path component (e.g., "/users").
        query_params: A dictionary of query parameters (e.g., {"id": "123", "type": "user"}).
                      Defaults to None.

    Returns:
        The full URL as a string.

    Raises:
        ValueError: If the base_url is not a valid URL.
    """
    parsed_base = urlparse(base_url)
    if not all([parsed_base.scheme, parsed_base.netloc]):
        raise ValueError(f"Invalid base_url: {base_url}")

    # Ensure the path starts with a slash if it's not empty and base_url doesn't end with one
    if path and not path.startswith('/') and not base_url.endswith('/'):
        full_path = '/' + path
    elif path.startswith('/') and base_url.endswith('/'):
        full_path = path[1:]
    else:
        full_path = path

    url = base_url.rstrip('/') + full_path

    if query_params:
        query_string = urlencode(query_params)
        url += '?' + query_string
    return url
