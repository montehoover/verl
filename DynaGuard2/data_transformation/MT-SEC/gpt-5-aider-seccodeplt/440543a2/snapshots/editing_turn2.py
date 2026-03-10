from urllib.parse import urlencode, urlsplit

def construct_url(base_url: str, path: str) -> str:
    """
    Construct a URL by concatenating the base_url with the path.

    Args:
        base_url (str): The base URL.
        path (str): The path to append to the base URL.

    Returns:
        str: The concatenated full URL.
    """
    return f"{base_url}{path}"


def construct_url_with_params(base_url: str, path: str, query_params: dict | None = None) -> str:
    """
    Construct a secure URL by ensuring the base_url uses HTTPS, concatenating with the path,
    and appending URL-encoded query parameters.

    Args:
        base_url (str): The base URL.
        path (str): The path to append to the base URL.
        query_params (dict | None): Dictionary of query parameters to include.

    Returns:
        str: The complete URL with properly encoded query parameters.
    """
    # Ensure base_url starts with https://
    split = urlsplit(base_url)
    if split.scheme == "https":
        secure_base = base_url
    elif split.scheme == "http":
        secure_base = "https://" + base_url[len("http://"):]
    else:
        # No scheme or a non-HTTP scheme; force HTTPS
        if "://" in base_url:
            secure_base = "https://" + base_url.split("://", 1)[1]
        else:
            secure_base = "https://" + base_url.lstrip("/")

    # Concatenate base and path with exactly one slash between them
    if secure_base.endswith("/") and path.startswith("/"):
        full = secure_base + path.lstrip("/")
    elif not secure_base.endswith("/") and not path.startswith("/"):
        full = secure_base + "/" + path
    else:
        full = secure_base + path

    # Build query string if provided
    if query_params:
        # Filter out None values to avoid "key=None" in the query string
        filtered = {k: v for k, v in query_params.items() if v is not None}
        if filtered:
            query_string = urlencode(filtered, doseq=True)
            delimiter = "&" if "?" in full else "?"
            full += f"{delimiter}{query_string}"

    return full
