import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def create_oauth_callback_url(base_url: str, callback_path: str, state: str) -> str:
    """
    Create a complete OAuth callback URL by combining the base URL with a callback path
    and appending a 'state' query parameter.

    Args:
        base_url: The base URL of the application (e.g., 'https://auth.myapp.com').
        callback_path: The path for the OAuth callback (e.g., '/oauth/callback').
        state: A unique state parameter for the OAuth flow.

    Returns:
        The complete OAuth callback URL as a string.

    Raises:
        ValueError: If the resulting callback URL is not from an allowed domain.
    """
    # Normalize base_url to include a scheme if missing
    parsed_base = urllib.parse.urlparse(base_url)
    if not parsed_base.scheme:
        base_url = 'https://' + base_url

    # Join base URL and callback path
    joined_url = urllib.parse.urljoin(base_url, callback_path)
    parsed = urllib.parse.urlparse(joined_url)

    # Validate the hostname against allowed domains
    hostname = parsed.hostname
    if not hostname or hostname not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError("Callback URL domain is not allowed")

    # Merge existing query params with the state parameter
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query_dict = dict(query_pairs)
    query_dict['state'] = state
    new_query = urllib.parse.urlencode(query_dict, doseq=True)

    # Reconstruct the final URL
    final_url = urllib.parse.urlunparse(parsed._replace(query=new_query))
    return final_url
