from urllib.parse import urlencode

def construct_url_with_params(base_url: str, path: str, query_params: dict = None) -> str:
    """
    Constructs a URL from a base URL, path, and optional query parameters,
    ensuring the URL uses HTTPS.

    Args:
        base_url: The base URL (e.g., "http://example.com" or "example.com").
        path: The path component (e.g., "/users").
        query_params: A dictionary of query parameters (e.g., {"id": "123", "type": "user"}).

    Returns:
        The full URL with query parameters, using HTTPS.
    """
    if not base_url.startswith("https://"):
        if base_url.startswith("http://"):
            base_url = "https://" + base_url[len("http://"):]
        else:
            base_url = "https://" + base_url

    url = base_url.rstrip('/') + '/' + path.lstrip('/')

    if query_params:
        query_string = urlencode(query_params)
        url += "?" + query_string

    return url
