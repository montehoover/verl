from urllib.parse import urlencode, urlparse

VALID_DOMAINS = ["example.com", "google.com", "bing.com"]

def construct_url_with_params(base_url: str, path: str, query_params: dict = None) -> str:
    """
    Constructs a URL by combining a base URL, a path, and optional query parameters.
    Ensures the base URL is from a valid domain.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path component (e.g., "/users").
        query_params: A dictionary of query parameters (e.g., {"id": "123", "type": "user"}).

    Returns:
        The full URL with query parameters.

    Raises:
        ValueError: If the base_url is not from a valid domain.
    """
    parsed_url = urlparse(base_url)
    if parsed_url.netloc not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain: {parsed_url.netloc}. Must be one of {VALID_DOMAINS}")

    # Ensure path starts with a slash if base_url doesn't end with one,
    # and path doesn't already start with one.
    # Also, ensure no double slashes if base_url ends with / and path starts with /.
    if base_url.endswith('/') and path.startswith('/'):
        url = base_url[:-1] + path
    elif not base_url.endswith('/') and not path.startswith('/'):
        url = base_url + '/' + path
    else:
        url = base_url + path

    if query_params:
        query_string = urlencode(query_params)
        url += "?" + query_string
    return url
