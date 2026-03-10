from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

def assemble_oauth_callback_url(root_url: str, path_for_callback: str, session_token: str) -> str:
    """
    Combine a root URL and a callback path into a single OAuth callback URL and
    append a 'state' query parameter with the provided session_token.

    Behavior:
    - Ensures exactly one '/' joins root and path when path is not absolute.
    - If path_for_callback is an absolute URL (http/https), root_url is ignored.
    - If root_url is empty and path is relative, ensures the path starts with '/'.
    - Preserves existing query parameters and fragments.
    - Sets or overrides the 'state' query parameter to session_token.
    """
    if not isinstance(root_url, str) or not isinstance(path_for_callback, str) or not isinstance(session_token, str):
        raise TypeError("root_url, path_for_callback, and session_token must all be strings.")

    root = root_url.strip()
    path = path_for_callback.strip()

    # Determine the base URL to which we'll append the 'state' parameter.
    lower_path = path.lower()
    if lower_path.startswith(("http://", "https://")):
        base_url = path
    else:
        if not root:
            base_url = path if path.startswith("/") else f"/{path}"
        else:
            base_url = root.rstrip("/") + "/" + path.lstrip("/")

    # Parse the URL, merge/override the 'state' parameter, and rebuild.
    parts = urlsplit(base_url)
    query_params = dict(parse_qsl(parts.query, keep_blank_values=True))
    query_params["state"] = session_token
    new_query = urlencode(query_params)

    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))
