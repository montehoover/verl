import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def assemble_oauth_callback(application_url: str, callback_route: str, token_state: str) -> str:
    """
    Constructs an OAuth callback URL and validates its domain.

    Args:
        application_url: The root URL for the application (e.g., "https://auth.myapp.com").
        callback_route: The endpoint for the OAuth callback handler (e.g., "/oauth/callback").
        token_state: A unique identifier for the OAuth exchange.

    Returns:
        The complete OAuth callback URL string.

    Raises:
        ValueError: If the domain of the application_url is not in ALLOWED_CALLBACK_DOMAINS.
    """
    parsed_app_url = urllib.parse.urlparse(application_url)
    if parsed_app_url.hostname not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(
            f"Domain '{parsed_app_url.hostname}' is not an allowed callback domain."
        )

    # Ensure application_url ends with a slash if not already present for proper joining
    if not application_url.endswith('/'):
        application_url += '/'
    
    # Ensure callback_route does not start with a slash if application_url ends with one
    # or ensure it does start with a slash if application_url does not (urljoin handles this)
    # For simplicity, we'll let urljoin handle the path joining logic.
    # We'll remove leading slash from callback_route to be safe with urljoin.
    full_path = urllib.parse.urljoin(application_url, callback_route.lstrip('/'))

    # Prepare query parameters
    query_params = {'state': token_state}
    query_string = urllib.parse.urlencode(query_params)

    # Construct the final URL
    final_url = f"{full_path}?{query_string}"
    
    return final_url
