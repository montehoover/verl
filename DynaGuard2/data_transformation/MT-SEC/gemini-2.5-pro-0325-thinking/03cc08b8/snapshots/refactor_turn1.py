import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def assemble_oauth_callback(application_url: str, callback_route: str, token_state: str) -> str:
    """
    Creates an OAuth callback URL typically used in third-party authentication workflows.
    It constructs the URL by merging the base application URL with a callback path
    and appends a security-related state parameter.

    Args:
        application_url: A string identifying the root URL for the application.
        callback_route: A string representing the endpoint for the OAuth callback handler.
        token_state: A string that provides a unique identifier to maintain the
                     integrity of the OAuth exchange.

    Returns:
        A string that contains the fully assembled OAuth callback URL.

    Raises:
        ValueError: Raised when the callback URL domain fails to meet the security
                    requirements of allowed or same-domain restrictions.
    """
    # Ensure callback_route starts with a slash if it's not empty and application_url doesn't end with one
    if callback_route and not callback_route.startswith('/') and not application_url.endswith('/'):
        effective_callback_route = '/' + callback_route
    elif callback_route.startswith('/') and application_url.endswith('/'):
        effective_callback_route = callback_route[1:]
    else:
        effective_callback_route = callback_route

    base_callback_url = urllib.parse.urljoin(application_url, effective_callback_route)
    
    parsed_app_url = urllib.parse.urlparse(application_url)
    app_domain = parsed_app_url.netloc.split(':')[0] # Remove port if present

    parsed_callback_url = urllib.parse.urlparse(base_callback_url)
    callback_domain = parsed_callback_url.netloc.split(':')[0] # Remove port if present

    if not (callback_domain == app_domain or callback_domain in ALLOWED_CALLBACK_DOMAINS):
        raise ValueError(
            f"Callback URL domain '{callback_domain}' is not allowed. "
            f"Must be '{app_domain}' or one of {ALLOWED_CALLBACK_DOMAINS}."
        )

    # Add the state parameter
    query_params = {'state': token_state}
    # urlencode quotes the state parameter appropriately
    final_url = urllib.parse.urlunparse(
        parsed_callback_url._replace(query=urllib.parse.urlencode(query_params))
    )
    
    return final_url
