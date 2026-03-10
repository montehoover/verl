from urllib.parse import urlparse, urlencode


def parse_and_validate_url(url: str) -> dict:
    """
    Parse and validate a URL string.

    Args:
        url (str): The URL to parse.

    Returns:
        dict: A dictionary with keys 'scheme', 'netloc', and 'path'.

    Raises:
        ValueError: If the URL is invalid.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    url = url.strip()
    if not url:
        raise ValueError("URL is empty")

    parsed = urlparse(url)

    if not parsed.scheme:
        raise ValueError("URL is missing a scheme")

    if parsed.scheme.lower() == "file":
        # file URLs may have empty netloc but must have a path
        if not parsed.path:
            raise ValueError("File URL must include a path")
    else:
        if not parsed.netloc:
            raise ValueError("URL is missing a network location (netloc/host)")

    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
    }


def generate_query_string(params: dict) -> str:
    """
    Generate a URL-encoded query string from a dictionary of parameters.

    - Omits keys with value None.
    - Expands list/tuple values into multiple key=value pairs (doseq=True).

    Args:
        params (dict): Dictionary of query parameters.

    Returns:
        str: URL-encoded query string (without leading '?').

    Raises:
        ValueError: If params is not a dictionary or contains nested dictionaries.
    """
    if params is None:
        return ""
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")

    filtered = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, dict):
            raise ValueError("Nested dictionaries are not supported in query parameters")
        filtered[key] = value

    if not filtered:
        return ""

    return urlencode(filtered, doseq=True)
