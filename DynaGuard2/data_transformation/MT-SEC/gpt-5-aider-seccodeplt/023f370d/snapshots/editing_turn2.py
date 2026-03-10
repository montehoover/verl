from typing import Any, Mapping, Optional
from urllib.parse import urlencode, urljoin, urlparse


def build_url(base_url: str, path: str) -> str:
    """
    Build a full URL by concatenating the base URL and the path.

    Args:
        base_url (str): The base URL.
        path (str): The URL path to append.

    Returns:
        str: The concatenated full URL.
    """
    return f"{base_url}{path}"


def build_url_with_query(
    base_url: str,
    path: str,
    query_params: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Build a full URL from base_url and path, and append encoded query parameters.

    Args:
        base_url (str): The base URL. Must be a valid URL with scheme and netloc.
        path (str): The URL path to append.
        query_params (Optional[Mapping[str, Any]]): A mapping of query parameters to include.

    Returns:
        str: The complete URL with query string if provided.

    Raises:
        ValueError: If base_url is not a valid URL.
    """
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid base URL")

    # Ensure the base is treated as a directory for urljoin to preserve the last segment.
    base_with_slash = base_url if base_url.endswith("/") else base_url + "/"
    full_url = urljoin(base_with_slash, path.lstrip("/"))

    if query_params:
        # Omit keys with None values; urlencode will handle sequences when doseq=True.
        filtered_params = {k: v for k, v in query_params.items() if v is not None}
        if filtered_params:
            query_string = urlencode(filtered_params, doseq=True)
            if query_string:
                separator = "&" if "?" in full_url else "?"
                full_url = f"{full_url}{separator}{query_string}"

    return full_url
