from urllib.parse import urlencode

def construct_url_with_params(base_url: str, path: str, query_params: dict = None) -> str:
    """
    Constructs a URL from a base URL, a path, and optional query parameters,
    ensuring the URL uses HTTPS.

    Args:
        base_url: The base URL (e.g., "http://example.com" or "example.com").
        path: The path component (e.g., "/users").
        query_params: A dictionary of query parameters (e.g., {"id": "123", "type": "user"}).

    Returns:
        The full URL with query parameters.
    """
    if not base_url.startswith("https://"):
        if base_url.startswith("http://"):
            base_url = "https://" + base_url[len("http://"):]
        else:
            base_url = "https://" + base_url

    # Ensure the path starts with a slash if it's not empty and base_url doesn't end with one
    if path and not path.startswith("/") and not base_url.endswith("/"):
        full_url = base_url + "/" + path
    # Prevent double slashes if base_url ends with / and path starts with /
    elif base_url.endswith("/") and path.startswith("/"):
        full_url = base_url + path[1:]
    else:
        full_url = base_url + path

    if query_params:
        query_string = urlencode(query_params)
        full_url += "?" + query_string
    
    return full_url
