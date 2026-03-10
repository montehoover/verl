from urllib.parse import urlparse, urlencode
import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def validate_domain(url):
    """
    Validates if the domain of the given URL is in the list of approved domains.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if the domain is approved
        
    Raises:
        ValueError: If the domain is not in the approved list
    """
    # List of approved domains
    APPROVED_DOMAINS = [
        'example.com',
        'trusted-site.org',
        'safe-domain.net',
        'approved.io'
    ]
    
    # Parse the URL to extract the domain
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    
    # Remove www. prefix if present
    if domain.startswith('www.'):
        domain = domain[4:]
    
    # Check if domain is in approved list
    if domain in APPROVED_DOMAINS:
        return True
    else:
        raise ValueError(f"Domain '{domain}' is not in the list of approved domains")


def create_query_string(params):
    """
    Creates a URL-encoded query string from a dictionary of parameters.
    
    Args:
        params (dict): Dictionary of parameters to encode
        
    Returns:
        str: URL-encoded query string
    """
    return urlencode(params)


def build_sso_url(sso_path, uid, redirect_url):
    """
    Builds a complete SSO redirect URL.
    
    Args:
        sso_path (str): The base SSO path
        uid (str): User ID
        redirect_url (str): The redirect URL after SSO
        
    Returns:
        str: Complete SSO redirect URL
        
    Raises:
        ValueError: If the redirect URL's domain is not in ALLOWED_RETURN_DOMAINS
    """
    # Parse the redirect URL to extract the domain
    parsed_redirect = urllib.parse.urlparse(redirect_url)
    redirect_domain = parsed_redirect.netloc.lower()
    
    # Remove www. prefix if present
    if redirect_domain.startswith('www.'):
        redirect_domain = redirect_domain[4:]
    
    # Check if domain is allowed
    if redirect_domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(f"Redirect domain '{redirect_domain}' is not allowed")
    
    # Build the SSO URL with query parameters
    params = {
        'uid': uid,
        'redirect_url': redirect_url
    }
    
    query_string = urllib.parse.urlencode(params)
    
    # Combine SSO path with query string
    if '?' in sso_path:
        return f"{sso_path}&{query_string}"
    else:
        return f"{sso_path}?{query_string}"
