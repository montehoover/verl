import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def assemble_oauth_callback_url(root_url: str, path_for_callback: str, session_token: str) -> str:
    """
    Builds an OAuth callback URL by combining a base URL, a callback path or absolute URL,
    and a state parameter to secure the flow.

    Args:
        root_url: The main URL of the application (absolute HTTP(S) URL).
        path_for_callback: A path (e.g., "/oauth/callback") to execute OAuth callbacks, or an absolute URL.
        session_token: A string used as the OAuth state parameter.

    Returns:
        The full OAuth callback URL as a string.

    Raises:
        ValueError: If inputs are invalid, or if the callback URL resolves to an invalid or unauthorized domain.
    """
    # Basic type validation (raise ValueError per spec)
    if not isinstance(root_url, str) or not isinstance(path_for_callback, str) or not isinstance(session_token, str):
        raise ValueError("root_url, path_for_callback, and session_token must all be strings")

    # Validate root_url is a proper absolute HTTP(S) URL
    parsed_root = urllib.parse.urlparse(root_url)
    if parsed_root.scheme not in ("http", "https") or not parsed_root.netloc:
        raise ValueError("root_url must be an absolute HTTP(S) URL")

    # Parse the callback path/URL
    parsed_callback = urllib.parse.urlparse(path_for_callback)

    # If an absolute URL is provided for the callback, validate against the allowlist
    if parsed_callback.scheme or parsed_callback.netloc:
        callback_host = parsed_callback.hostname
        if not callback_host or callback_host not in ALLOWED_CALLBACK_DOMAINS:
            raise ValueError("Unauthorized callback domain")
        base_callback_url = path_for_callback
    else:
        # Treat as a path, join with the root_url
        normalized_root = root_url.rstrip("/") + "/"
        # Default to "/" if empty path provided
        path_fragment = path_for_callback if path_for_callback else "/"
        base_callback_url = urllib.parse.urljoin(normalized_root, path_fragment)

    # Parse the result to add/merge the state parameter
    final_parts = urllib.parse.urlparse(base_callback_url)

    # If a relative path was provided, ensure the resolved host didn't change
    if not (parsed_callback.scheme or parsed_callback.netloc):
        if final_parts.hostname != parsed_root.hostname:
            raise ValueError("Callback URL resolved outside the provided root_url domain")

    # Merge query parameters and set/override the 'state' parameter
    query_dict = urllib.parse.parse_qs(final_parts.query, keep_blank_values=True)
    query_dict["state"] = [session_token]
    new_query = urllib.parse.urlencode(query_dict, doseq=True)

    # Rebuild the final URL with the updated query
    final_url = urllib.parse.urlunparse(final_parts._replace(query=new_query))
    return final_url
