from urllib.parse import urlparse, urljoin, quote
from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def validate_url(url):
    """
    Validates a URL string.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if the URL is valid
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url:
        raise ValueError("URL cannot be empty")
    
    try:
        result = urlparse.urlparse(url)
        
        # Check if scheme and netloc are present
        if not result.scheme:
            raise ValueError("URL must include a scheme (e.g., http://, https://)")
        
        if not result.netloc:
            raise ValueError("URL must include a domain")
        
        # Check for valid schemes
        valid_schemes = ['http', 'https', 'ftp', 'ftps']
        if result.scheme.lower() not in valid_schemes:
            raise ValueError(f"Invalid URL scheme. Must be one of: {', '.join(valid_schemes)}")
        
        return True
        
    except Exception as e:
        raise ValueError(f"Invalid URL: {str(e)}")


def concatenate_url_path(base_url, path):
    """
    Concatenates a base URL with a path.
    
    Args:
        base_url (str): The base URL
        path (str): The path to append
        
    Returns:
        str: The concatenated URL
    """
    # Remove leading slash from path if present to avoid double slashes
    if path.startswith('/'):
        path = path[1:]
    
    # Ensure base_url ends with a slash
    if not base_url.endswith('/'):
        base_url = base_url + '/'
    
    # Use urljoin to properly concatenate
    result = urljoin(base_url, path)
    
    return result


def construct_redirect_link(domain_base_url, next_redirect_param):
    """
    Constructs a secure redirect URL after user login.
    
    Args:
        domain_base_url (str): The base URL of the domain
        next_redirect_param (str): The redirect URL parameter
        
    Returns:
        str: The complete redirect URL
        
    Raises:
        ValueError: If the redirect URL is not within the base domain or allowed external domains
    """
    # Parse the base domain URL
    base_parsed = urlparse.urlparse(domain_base_url)
    base_domain = base_parsed.netloc
    
    # Parse the redirect parameter
    redirect_parsed = urlparse.urlparse(next_redirect_param)
    
    # If redirect is a relative path, construct full URL with base domain
    if not redirect_parsed.scheme and not redirect_parsed.netloc:
        # It's a relative path, safe to use
        return urljoin(domain_base_url, next_redirect_param)
    
    # If it's an absolute URL, check if it's allowed
    redirect_domain = redirect_parsed.netloc
    
    # Check if redirect domain matches base domain
    if redirect_domain == base_domain:
        return next_redirect_param
    
    # Check if redirect domain is in allowed external domains
    if redirect_domain in ALLOWED_DOMAINS:
        return next_redirect_param
    
    # If none of the above, raise error
    raise ValueError(f"Redirect URL domain '{redirect_domain}' is not allowed. Must be within '{base_domain}' or one of: {', '.join(ALLOWED_DOMAINS)}")
