import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def generate_oauth_url(host_url: str, path_callback: str, session_id: str) -> str:
    """
    Creates an OAuth callback URL.

    Constructs the URL by merging the base application URL with a callback path
    and appends a security-related state parameter.

    Args:
        host_url: The root URL for the application.
        path_callback: The endpoint for the OAuth callback handler.
        session_id: A unique identifier to maintain the integrity of the OAuth exchange.

    Returns:
        The fully assembled OAuth callback URL.

    Raises:
        ValueError: If the callback URL domain fails to meet security requirements.
    """
    parsed_host_url = urllib.parse.urlparse(host_url)
    host_domain = parsed_host_url.hostname

    if host_domain not in ALLOWED_CALLBACK_DOMAINS:
        # Check if the callback path itself is a full URL with a different domain
        parsed_callback_path = urllib.parse.urlparse(path_callback)
        if parsed_callback_path.hostname and parsed_callback_path.hostname not in ALLOWED_CALLBACK_DOMAINS:
            raise ValueError(
                f"Callback domain '{parsed_callback_path.hostname}' is not an allowed domain."
            )
        elif not parsed_callback_path.hostname and host_domain not in ALLOWED_CALLBACK_DOMAINS: # path_callback is a relative path
             raise ValueError(
                f"Host domain '{host_domain}' for callback is not an allowed domain."
            )


    # Ensure path_callback starts with a slash if it's a relative path
    if not path_callback.startswith('/') and not urllib.parse.urlparse(path_callback).scheme:
        path_callback = '/' + path_callback

    # Construct the base callback URL
    callback_base_url = urllib.parse.urljoin(host_url, path_callback)

    # Add the state parameter
    url_parts = list(urllib.parse.urlparse(callback_base_url))
    query = urllib.parse.parse_qs(url_parts[4])
    query['state'] = [session_id]
    url_parts[4] = urllib.parse.urlencode(query, doseq=True)

    return urllib.parse.urlunparse(url_parts)
