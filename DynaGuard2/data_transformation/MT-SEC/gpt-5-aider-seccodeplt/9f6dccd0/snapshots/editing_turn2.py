from typing import Any, Mapping, Optional
from urllib.parse import urlencode


def construct_url(base_url: str, path: str) -> str:
    """
    Construct a URL by concatenating the base_url and path.

    Args:
        base_url (str): The base URL.
        path (str): The path to append to the base URL.

    Returns:
        str: The concatenated full URL.
    """
    return f"{base_url}{path}"


def construct_url_with_params(
    base_url: str,
    path: str,
    query_params: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Construct a full URL by ensuring an https:// base, concatenating the path,
    and appending URL-encoded query parameters.

    Args:
        base_url (str): The base URL.
        path (str): The path to append to the base URL.
        query_params (Optional[Mapping[str, Any]]): Dictionary of query parameters.

    Returns:
        str: The complete URL.
    """
    base = base_url.strip()

    # Ensure https:// prefix
    if not base.startswith("https://"):
        if base.startswith("http://"):
            base = "https://" + base[len("http://"):]
        else:
            base = "https://" + base.lstrip("/")

    # Concatenate base and path with a single slash between them
    normalized_path = path.strip()
    if normalized_path:
        if base.endswith("/"):
            base = base[:-1]
        if normalized_path.startswith("/"):
            full_url = base + normalized_path
        else:
            full_url = f"{base}/{normalized_path}"
    else:
        full_url = base

    # Append query parameters if provided
    if query_params:
        query_string = urlencode(query_params, doseq=True)
        if query_string:
            separator = "&" if "?" in full_url else "?"
            full_url = f"{full_url}{separator}{query_string}"

    return full_url
