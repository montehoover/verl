import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def construct_oauth_callback_uri(root_url: str, callback_uri: str, session_state: str) -> str:
    """
    Constructs an OAuth callback URI, validates its domain, and appends session state.

    Args:
        root_url: The application's main base URL (e.g., "https://auth.myapp.com").
        callback_uri: The specific path for the OAuth callback endpoint (e.g., "/oauth/callback").
        session_state: A unique value for ensuring the integrity of the OAuth request flow.

    Returns:
        The finalized OAuth callback URL as a string.

    Raises:
        ValueError: If the domain of the constructed callback URL is not in ALLOWED_CALLBACK_DOMAINS.
    """
    # Ensure root_url ends with a slash for proper joining if callback_uri is relative
    if not root_url.endswith('/'):
        root_url += '/'

    # Join root_url and callback_uri
    # urllib.parse.urljoin handles cases where callback_uri might be absolute or relative
    intermediate_url = urllib.parse.urljoin(root_url, callback_uri)

    # Parse the intermediate URL to extract its components, especially the netloc (domain)
    parsed_intermediate_url = urllib.parse.urlparse(intermediate_url)
    domain = parsed_intermediate_url.netloc

    # Validate the domain
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(
            f"Domain '{domain}' is not an allowed callback domain. "
            f"Allowed domains are: {ALLOWED_CALLBACK_DOMAINS}"
        )

    # Add session_state as a query parameter
    # We need to handle existing query parameters in callback_uri if any, though typically it's a path.
    # For simplicity, we assume callback_uri is a path and session_state is the primary query param.
    # If callback_uri could contain query params, a more robust query builder would be needed.
    
    # Construct query parameters
    query_params = {'state': session_state}
    
    # Reconstruct the URL with the new query parameter
    # This approach correctly handles if intermediate_url already has query parameters or not.
    url_parts = list(parsed_intermediate_url) # Convert ParseResult to a mutable list
    
    # Current query as a dictionary
    current_query = urllib.parse.parse_qs(url_parts[4]) # url_parts[4] is the query string
    current_query.update(query_params) # Add/update our session_state
    
    # Encode the updated query parameters back into a string
    url_parts[4] = urllib.parse.urlencode(current_query, doseq=True)
    
    final_url = urllib.parse.urlunparse(url_parts)
    
    return final_url
