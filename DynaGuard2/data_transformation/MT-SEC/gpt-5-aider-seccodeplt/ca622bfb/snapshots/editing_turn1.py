def assemble_oauth_callback_url(root_url: str, path_for_callback: str) -> str:
    """
    Combine a root URL and a callback path into a single OAuth callback URL.

    - Ensures exactly one '/' joins root and path.
    - If path_for_callback is an absolute URL (http/https), it is returned as-is.
    - If root_url is empty, returns the path, ensuring it starts with '/' (unless absolute).
    """
    if not isinstance(root_url, str) or not isinstance(path_for_callback, str):
        raise TypeError("Both root_url and path_for_callback must be strings.")

    root = root_url.strip()
    path = path_for_callback.strip()

    # If the path is already an absolute URL, return it as-is.
    if path.lower().startswith(("http://", "https://")):
        return path

    # If root is empty, normalize and return the path.
    if not root:
        return path if path.startswith("/") else f"/{path}"

    # Normalize and join with exactly one slash.
    root = root.rstrip("/")
    path = "/" + path.lstrip("/")

    return root + path
