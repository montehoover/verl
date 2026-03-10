from urllib.parse import urlparse, urljoin
from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def validate_url(url):
    """
    Validates if a URL is well-formed.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if the URL is well-formed
        
    Raises:
        ValueError: If the URL is not well-formed
    """
    try:
        result = urlparse(url)
        # Check if scheme and netloc are present (minimum requirements for a valid URL)
        if not result.scheme or not result.netloc:
            raise ValueError(f"Invalid URL: '{url}' - missing scheme or network location")
        
        # Check if scheme is valid (http, https, ftp, etc.)
        valid_schemes = ['http', 'https', 'ftp', 'ftps', 'file']
        if result.scheme.lower() not in valid_schemes:
            raise ValueError(f"Invalid URL: '{url}' - unsupported scheme '{result.scheme}'")
            
        return True
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Invalid URL: '{url}' - {str(e)}")


def concatenate_url_path(base_url, path):
    """
    Concatenates a base URL with a path to create a full URL.
    
    Args:
        base_url (str): The base URL
        path (str): The path to append
        
    Returns:
        str: The full URL with the path appended
    """
    # Remove trailing slashes from base_url to avoid double slashes
    base_url = base_url.rstrip('/')
    
    # Ensure path starts with a slash for proper joining
    if path and not path.startswith('/'):
        path = '/' + path
    
    # Use urljoin to properly concatenate the URL and path
    return urljoin(base_url, path)


def generate_redirect_link(app_base_url, redirect_param):
    """
    Generates a secure redirect URL after login.
    
    Args:
        app_base_url (str): The base URL of the application
        redirect_param (str): The redirect URL parameter
        
    Returns:
        str: The complete redirect URL
        
    Raises:
        ValueError: If the redirect URL is not within the base domain or allowed external domains
    """
    # Parse the base URL to get its domain
    base_parsed = urlparse.urlparse(app_base_url)
    base_domain = base_parsed.netloc.lower()
    
    # Parse the redirect parameter
    redirect_parsed = urlparse.urlparse(redirect_param)
    redirect_domain = redirect_parsed.netloc.lower()
    
    # If redirect_param is a relative path, join it with the base URL
    if not redirect_parsed.scheme and not redirect_parsed.netloc:
        return urljoin(app_base_url, redirect_param)
    
    # Check if the redirect domain matches the base domain
    if redirect_domain == base_domain:
        return redirect_param
    
    # Check if the redirect domain is in the allowed domains list
    if redirect_domain in ALLOWED_DOMAINS:
        return redirect_param
    
    # If none of the above conditions are met, raise ValueError
    raise ValueError(f"Redirect URL domain '{redirect_domain}' is not allowed")
