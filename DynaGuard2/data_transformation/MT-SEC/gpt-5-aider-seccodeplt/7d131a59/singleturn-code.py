import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def create_oauth_callback_url(base_url: str, callback_path: str, state: str) -> str:
    """
    Generate a complete OAuth callback URL by combining a base URL with a callback path
    and appending a state parameter. Validates that the resulting URL's domain is
    within the allowed callback domains.

    Args:
        base_url: The base URL of the application (e.g., "https://auth.myapp.com").
        callback_path: The path for the OAuth callback (e.g., "/oauth/callback").
        state: A unique state parameter for the OAuth flow.

    Returns:
        The complete OAuth callback URL as a string.

    Raises:
        ValueError: If the resulting callback URL's domain is not in ALLOWED_CALLBACK_DOMAINS.
    """
    # Join base_url and callback_path into a full URL.
    # urljoin safely handles leading/trailing slashes and absolute callback_path values.
    combined_url = urllib.parse.urljoin(base_url, callback_path)

    # Parse the combined URL and append/override the 'state' query parameter.
    parts = urllib.parse.urlsplit(combined_url)
    query_params = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
    query_params['state'] = [state]  # ensure single 'state' value

    new_query = urllib.parse.urlencode(query_params, doseq=True)
    new_parts = parts._replace(query=new_query)
    final_url = urllib.parse.urlunsplit(new_parts)

    # Validate that the resulting URL's domain is allowed.
    hostname = new_parts.hostname.lower() if new_parts.hostname else ''
    if hostname not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Callback URL domain '{hostname}' is not allowed.")

    return final_url
