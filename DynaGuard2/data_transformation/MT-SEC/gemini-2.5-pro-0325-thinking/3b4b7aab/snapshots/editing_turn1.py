from urllib.parse import urljoin

def build_url_with_query(base_url: str, path: str, query_param: str) -> str:
    """
    Constructs a URL by combining a base URL with a path and a query parameter.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path segment (e.g., "api/users").
        query_param: The query parameter string (e.g., "id=123").

    Returns:
        The full URL as a string.
    """
    # Ensure base_url ends with a slash for proper joining
    if not base_url.endswith('/'):
        base_url += '/'
    
    # Ensure path does not start with a slash if base_url ends with one
    # or urljoin might interpret path as an absolute path.
    # However, urljoin handles this correctly by replacing the last component of base_url's path.
    # For simplicity and robustness, we'll let urljoin handle it.
    # If path starts with '/', urljoin will treat it relative to the domain of base_url.
    # If path does not start with '/', it's appended to base_url's path.

    full_path = urljoin(base_url, path)
    
    # Append the query parameter
    if '?' in full_path:
        # If the joined path already has query parameters (less likely with simple path join)
        url_with_query = f"{full_path}&{query_param}"
    else:
        url_with_query = f"{full_path}?{query_param}"
        
    return url_with_query
