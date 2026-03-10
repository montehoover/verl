from urllib.parse import urlencode, urljoin

def build_url_with_query(base_url: str, path: str, query_params: dict) -> str:
    """
    Constructs a URL from a base URL, a path, and query parameters.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path component of the URL (e.g., "/api/users").
        query_params: A dictionary of query parameters (e.g., {"id": 123, "name": "test"}).

    Returns:
        The complete URL string with query parameters.
    """
    # Ensure base_url ends with a slash and path doesn't start with one for urljoin
    if not base_url.endswith('/'):
        base_url += '/'
    
    # urljoin handles joining base_url and path correctly
    # lstrip('/') from path to avoid issues if path starts with '/'
    url_with_path = urljoin(base_url, path.lstrip('/'))

    if query_params:
        query_string = urlencode(query_params)
        return f"{url_with_path}?{query_string}"
    else:
        return url_with_path
