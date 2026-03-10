from urllib.parse import urlparse


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
