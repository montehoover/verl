def create_oauth_callback_url(base_url: str, callback_path: str) -> str:
    """
    Build a full OAuth callback URL by combining a base URL and a callback path.

    - Ensures exactly one "/" between base_url and callback_path.
    - If callback_path is empty or None, returns base_url.
    - If base_url is empty or None, returns callback_path normalized to start with "/".
    """
    if not base_url and not callback_path:
        return ""
    if not base_url:
        return "/" + str(callback_path).lstrip("/")
    if not callback_path:
        return str(base_url)

    base = str(base_url).rstrip("/")
    path = str(callback_path).lstrip("/")

    return f"{base}/{path}"
