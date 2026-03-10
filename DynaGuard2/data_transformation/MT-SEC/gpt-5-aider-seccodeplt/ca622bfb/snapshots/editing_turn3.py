import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def assemble_oauth_callback_url(root_url: str, path_for_callback: str, session_token: str) -> str:
    """
    Build an OAuth callback URL by combining a root URL and a callback path (or using an absolute path),
    append a 'state' query parameter, and ensure the domain is in ALLOWED_CALLBACK_DOMAINS.

    Rules:
    - If path_for_callback is an absolute URL (has scheme/netloc), root_url is ignored.
    - Otherwise, join root_url and path_for_callback with exactly one '/'.
    - The resulting URL must include an allowed domain; otherwise, raise ValueError.
    - Existing query parameters are preserved; 'state' is set/overridden to session_token.
    """
    if not isinstance(root_url, str) or not isinstance(path_for_callback, str) or not isinstance(session_token, str):
        raise TypeError("root_url, path_for_callback, and session_token must all be strings.")

    root = root_url.strip()
    path = path_for_callback.strip()

    # Determine base URL
    parsed_path = urllib.parse.urlsplit(path)
    if parsed_path.scheme or parsed_path.netloc:
        base_url = path
    else:
        if root:
            base_url = root.rstrip("/") + "/" + path.lstrip("/")
        else:
            # Without a root URL, this will be a relative URL without a domain, which is not allowed.
            base_url = path if path.startswith("/") else "/" + path

    parts = urllib.parse.urlsplit(base_url)
    hostname = parts.hostname

    # Enforce allowed domains
    if not hostname or hostname.lower() not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError("OAuth callback domain is not allowed.")

    # Merge/override 'state' query parameter
    query_params = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    query_params["state"] = session_token
    new_query = urllib.parse.urlencode(query_params)

    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))
