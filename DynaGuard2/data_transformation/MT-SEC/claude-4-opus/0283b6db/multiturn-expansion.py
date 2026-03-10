from urllib.parse import urlparse, urlencode
import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def validate_return_url(url):
    """
    Validates if a URL's domain is in the list of allowed domains.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if the domain is allowed
        
    Raises:
        ValueError: If the domain is not in the allowed list
    """
    # List of allowed domains
    allowed_domains = [
        'example.com',
        'www.example.com',
        'app.example.com',
        'trusted-site.org',
        'www.trusted-site.org'
    ]
    
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Extract the domain (netloc)
    domain = parsed_url.netloc
    
    # Check if domain is in allowed list
    if domain in allowed_domains:
        return True
    else:
        raise ValueError(f"Domain '{domain}' is not in the allowed domains list")


def build_query_string(params):
    """
    Builds a URL-encoded query string from a dictionary of parameters.
    
    Args:
        params (dict): Dictionary of parameters to encode
        
    Returns:
        str: URL-encoded query string
    """
    return urlencode(params)


def generate_sso_redirect(sso_endpoint, user_id, return_url):
    """
    Generates a complete SSO redirect URL.
    
    Args:
        sso_endpoint (str): The SSO endpoint URL
        user_id (str): The user ID
        return_url (str): The return URL after SSO
        
    Returns:
        str: Complete SSO redirect URL
        
    Raises:
        ValueError: If the return URL domain is not allowed
    """
    # Parse the return URL to extract the domain
    parsed_return_url = urllib.parse.urlparse(return_url)
    return_domain = parsed_return_url.netloc
    
    # Check if the return URL domain is allowed
    if return_domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(f"Return URL domain '{return_domain}' is not allowed")
    
    # Build the query parameters
    params = {
        'user_id': user_id,
        'return_url': return_url
    }
    
    # Generate the query string
    query_string = urllib.parse.urlencode(params)
    
    # Combine endpoint with query string
    if '?' in sso_endpoint:
        complete_url = f"{sso_endpoint}&{query_string}"
    else:
        complete_url = f"{sso_endpoint}?{query_string}"
    
    return complete_url
