from urllib.parse import urlparse, urljoin
import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def validate_url_domain(url):
    """
    Validates if a URL's domain is in the list of allowed domains.
    
    Args:
        url (str): The URL to validate
        
    Raises:
        ValueError: If the domain is not in the allowed list
    """
    # List of allowed domains
    allowed_domains = [
        'example.com',
        'api.example.com',
        'staging.example.com',
        'localhost',
        '127.0.0.1'
    ]
    
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Extract the domain (netloc)
    domain = parsed_url.netloc.lower()
    
    # Remove port number if present
    if ':' in domain:
        domain = domain.split(':')[0]
    
    # Check if domain is in allowed list
    if domain not in allowed_domains:
        raise ValueError(f"Domain '{domain}' is not in the allowed domains list")
    
    return True


def combine_base_with_path(base, path):
    """
    Combines a base URL with a path, handling leading/trailing slashes gracefully.
    
    Args:
        base (str): The base URL
        path (str): The path to append
        
    Returns:
        str: The combined full URL
    """
    # Use urljoin to properly combine base and path
    # It handles slashes correctly
    return urljoin(base, path)


def build_oauth_redirect_url(main_url, redirect_path, nonce):
    """
    Creates OAuth callback URLs with validation and security.
    
    Args:
        main_url (str): The main/base URL
        redirect_path (str): The path for the OAuth callback
        nonce (str): Security nonce to include in the callback URL
        
    Returns:
        str: The complete OAuth callback URL
        
    Raises:
        ValueError: If the callback domain is not authorized
    """
    # Combine the main URL with the redirect path
    callback_url = urllib.parse.urljoin(main_url, redirect_path)
    
    # Parse the callback URL to extract the domain
    parsed_url = urllib.parse.urlparse(callback_url)
    domain = parsed_url.netloc.lower()
    
    # Remove port number if present
    if ':' in domain:
        domain = domain.split(':')[0]
    
    # Verify the domain is authorized
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not authorized for OAuth callbacks")
    
    # Add the nonce as a query parameter
    params = urllib.parse.parse_qs(parsed_url.query)
    params['nonce'] = [nonce]
    
    # Rebuild the URL with the nonce
    new_query = urllib.parse.urlencode(params, doseq=True)
    final_url = urllib.parse.urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        new_query,
        parsed_url.fragment
    ))
    
    return final_url
