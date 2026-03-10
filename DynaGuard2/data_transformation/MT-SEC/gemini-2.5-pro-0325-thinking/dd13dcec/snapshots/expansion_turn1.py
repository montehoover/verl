from urllib.parse import urlparse

def extract_url_parts(url_string):
    """
    Breaks down a URL into its components.

    Args:
        url_string: The URL to parse.

    Returns:
        A tuple containing the scheme, netloc, path, params, query, and fragment.
    """
    parsed_url = urlparse(url_string)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )
