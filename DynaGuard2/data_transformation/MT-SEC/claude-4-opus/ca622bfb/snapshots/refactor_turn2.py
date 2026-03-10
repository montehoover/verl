import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def extract_domain(url):
    """Extract domain from a URL."""
    parsed_url = urllib.parse.urlparse(url)
    return parsed_url.netloc

def validate_domain(domain, allowed_domains):
    """Validate if domain is in the allowed set."""
    if domain not in allowed_domains:
        raise ValueError(f"Domain '{domain}' is not authorized for OAuth callbacks")
    return domain

def normalize_path(path):
    """Ensure path starts with a slash."""
    if not path.startswith('/'):
        return '/' + path
    return path

def build_base_callback_url(root_url, path):
    """Combine root URL with callback path."""
    return urllib.parse.urljoin(root_url, path)

def add_state_parameter(url, state_value):
    """Add state parameter to URL query string."""
    parsed_url = urllib.parse.urlparse(url)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    query_params['state'] = [state_value]
    
    new_query = urllib.parse.urlencode(query_params, doseq=True)
    
    return urllib.parse.urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        new_query,
        parsed_url.fragment
    ))

def assemble_oauth_callback_url(root_url, path_for_callback, session_token):
    """
    Builds an OAuth callback URL for third-party authentication.
    
    Args:
        root_url: a string indicating the main URL of the application.
        path_for_callback: a string specifying the path to execute OAuth callbacks.
        session_token: a string used to pass a state parameter for verifying the OAuth transaction.
    
    Returns:
        Returns the full OAuth callback URL as a string.
    
    Raises:
        ValueError: raised if the callback URL points to an invalid or unauthorized domain.
    """
    # Pipeline of operations
    domain = extract_domain(root_url)
    validate_domain(domain, ALLOWED_CALLBACK_DOMAINS)
    normalized_path = normalize_path(path_for_callback)
    base_callback_url = build_base_callback_url(root_url, normalized_path)
    final_url = add_state_parameter(base_callback_url, session_token)
    
    return final_url
