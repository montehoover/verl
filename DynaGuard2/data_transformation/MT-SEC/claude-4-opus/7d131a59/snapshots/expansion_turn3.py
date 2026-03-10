from urllib.parse import urlparse, urlencode, urljoin
import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def parse_and_validate_url(url):
    """
    Parse and validate a URL string.
    
    Args:
        url (str): The URL string to parse and validate
        
    Returns:
        tuple: A tuple containing (scheme, netloc, path)
        
    Raises:
        ValueError: If the URL is invalid or missing required components
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url.strip():
        raise ValueError("URL cannot be empty")
    
    parsed = urlparse(url)
    
    # Check if scheme and netloc are present (minimum requirements for a valid URL)
    if not parsed.scheme:
        raise ValueError("URL must include a scheme (e.g., http, https)")
    
    if not parsed.netloc:
        raise ValueError("URL must include a network location (e.g., domain name)")
    
    # Return the required components
    return (parsed.scheme, parsed.netloc, parsed.path)


def build_url_with_query(base, path, params):
    """
    Build a URL with query parameters.
    
    Args:
        base (str): The base URL (e.g., "https://example.com")
        path (str): The path to append to the base URL
        params (dict): Dictionary of query parameters
        
    Returns:
        str: The complete URL with encoded query parameters
    """
    # Join base and path
    url = urljoin(base, path)
    
    # Add query parameters if provided
    if params:
        query_string = urlencode(params)
        # Add ? or & depending on whether URL already has query parameters
        if '?' in url:
            url = f"{url}&{query_string}"
        else:
            url = f"{url}?{query_string}"
    
    return url


def create_oauth_callback_url(base_url, callback_path, state):
    """
    Generate a secure OAuth callback URL.
    
    Args:
        base_url (str): The base URL for the callback
        callback_path (str): The path for the callback endpoint
        state (str): The state parameter for OAuth security
        
    Returns:
        str: The full OAuth callback URL
        
    Raises:
        ValueError: If the base URL domain is not in the allowed list
    """
    # Parse the base URL to extract the domain
    parsed_base = urllib.parse.urlparse(base_url)
    domain = parsed_base.netloc
    
    # Check if the domain is allowed
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the allowed callback domains")
    
    # Build the full callback URL with the state parameter
    params = {'state': state}
    callback_url = build_url_with_query(base_url, callback_path, params)
    
    return callback_url
