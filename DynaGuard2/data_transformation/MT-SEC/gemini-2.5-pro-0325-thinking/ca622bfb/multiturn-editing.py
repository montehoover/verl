import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def assemble_oauth_callback_url(root_url: str, path_for_callback: str, session_token: str) -> str:
    """
    Assembles an OAuth callback URL from a root URL, a callback path,
    and a session token, including the token as a 'state' query parameter.
    Raises ValueError if the root_url's domain is not in ALLOWED_CALLBACK_DOMAINS.

    Args:
        root_url: The base URL (e.g., "https://example.com").
        path_for_callback: The path for the callback (e.g., "/oauth/callback").
        session_token: The session token to be included as the 'state' parameter.

    Returns:
        The combined OAuth callback URL with the state parameter as a string.
    
    Raises:
        ValueError: If the domain of root_url is not in ALLOWED_CALLBACK_DOMAINS.
    """
    parsed_root_url = urllib.parse.urlparse(root_url)
    domain = parsed_root_url.netloc
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not an allowed callback domain.")

    base_callback_url = urllib.parse.urljoin(root_url, path_for_callback)
    
    # Prepare query parameters
    query_params = {'state': session_token}
    encoded_query_params = urllib.parse.urlencode(query_params)
    
    # Parse the base callback URL to append query parameters
    url_parts = list(urllib.parse.urlparse(base_callback_url))
    
    # Append new query parameters
    # If there are existing query parameters, append with '&', otherwise start with '?'
    if url_parts[4]:  # Check if query part is not empty
        url_parts[4] = f"{url_parts[4]}&{encoded_query_params}"
    else:
        url_parts[4] = encoded_query_params
        
    return urllib.parse.urlunparse(url_parts)
