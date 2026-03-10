from urllib.parse import urlparse, urljoin

def validate_url(url: str) -> bool:
    """
    Validates if the given string is a well-formed URL.

    Args:
        url: The string to validate.

    Returns:
        True if the URL is well-formed.

    Raises:
        ValueError: If the URL is not well-formed.
    """
    try:
        result = urlparse(url)
        # A well-formed URL must have a scheme (e.g., http, https) and a netloc (e.g., www.example.com)
        if all([result.scheme, result.netloc]):
            return True
        else:
            raise ValueError(f"URL is not well-formed: {url}")
    except Exception as e: # Catch any parsing errors as well
        raise ValueError(f"URL is not well-formed: {url}. Error: {e}")


def concatenate_url_path(base_url: str, path: str) -> str:
    """
    Concatenates a base URL and a path to form a full URL.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path to append (e.g., "/users/1" or "users/1").

    Returns:
        The full URL.
    """
    return urljoin(base_url, path)
