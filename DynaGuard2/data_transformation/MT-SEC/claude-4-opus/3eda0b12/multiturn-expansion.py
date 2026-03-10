from urllib.parse import urlparse, urljoin, urlunparse
from urllib import parse as urlparse

# List of permitted domains
PERMITTED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def validate_url(url):
    """
    Validates a URL and returns True if valid, raises ValueError if invalid.
    
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
        
        # Check if scheme and netloc are present (minimum requirements for a valid URL)
        if not result.scheme:
            raise ValueError("URL must have a scheme (e.g., http, https)")
        
        if not result.netloc:
            raise ValueError("URL must have a network location (domain)")
        
        # Check for valid schemes
        valid_schemes = ['http', 'https', 'ftp', 'ftps']
        if result.scheme.lower() not in valid_schemes:
            raise ValueError(f"URL scheme must be one of: {', '.join(valid_schemes)}")
        
        return True
        
    except Exception as e:
        raise ValueError(f"Invalid URL: {str(e)}")


def is_domain_permitted(url):
    """
    Checks if a URL's domain is in the list of permitted domains.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the domain is permitted, False otherwise
    """
    try:
        # First validate the URL
        validate_url(url)
        
        # Parse the URL to extract the domain
        parsed_url = urlparse.urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Remove port number if present
        if ':' in domain:
            domain = domain.split(':')[0]
        
        # Check if domain is in permitted list
        return domain in [d.lower() for d in PERMITTED_DOMAINS]
        
    except ValueError:
        # If URL is invalid, domain is not permitted
        return False


def construct_redirect_url(main_url, target_param):
    """
    Constructs a secure redirect URL by combining main_url and target_param.
    
    Args:
        main_url (str): The base URL
        target_param (str): The target parameter to append or combine
        
    Returns:
        str: The complete redirect URL
        
    Raises:
        ValueError: If the final URL is not within permitted domains
    """
    if not isinstance(main_url, str) or not isinstance(target_param, str):
        raise ValueError("Both main_url and target_param must be strings")
    
    # Parse the main URL
    try:
        parsed_main = urlparse.urlparse(main_url)
        if not parsed_main.scheme or not parsed_main.netloc:
            raise ValueError("main_url must be a complete URL with scheme and domain")
    except Exception as e:
        raise ValueError(f"Invalid main_url: {str(e)}")
    
    # Construct the redirect URL
    if target_param.startswith('http://') or target_param.startswith('https://'):
        # target_param is a full URL
        redirect_url = target_param
    elif target_param.startswith('/'):
        # target_param is an absolute path
        redirect_url = urlparse.urlunparse((
            parsed_main.scheme,
            parsed_main.netloc,
            target_param,
            '',
            '',
            ''
        ))
    else:
        # target_param is a relative path
        redirect_url = urlparse.urljoin(main_url, target_param)
    
    # Parse the final redirect URL
    parsed_redirect = urlparse.urlparse(redirect_url)
    redirect_domain = parsed_redirect.netloc.lower()
    
    # Remove port number if present
    if ':' in redirect_domain:
        redirect_domain = redirect_domain.split(':')[0]
    
    # Get main URL domain
    main_domain = parsed_main.netloc.lower()
    if ':' in main_domain:
        main_domain = main_domain.split(':')[0]
    
    # Check if redirect URL is within application domain or permitted domains
    if redirect_domain != main_domain and redirect_domain not in [d.lower() for d in PERMITTED_DOMAINS]:
        raise ValueError(f"Redirect URL domain '{redirect_domain}' is not permitted")
    
    return redirect_url
