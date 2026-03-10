import urllib.parse
import logging

# Configure logger
logger = logging.getLogger(__name__)

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def validate_callback_domain(domain):
    """
    Validates if a domain is in the allowed callback domains.
    
    Args:
        domain: The domain to validate.
    
    Returns:
        bool: True if domain is valid, False otherwise.
    """
    return domain in ALLOWED_CALLBACK_DOMAINS

def construct_oauth_url(scheme, netloc, path, state):
    """
    Constructs an OAuth URL with the given components and state parameter.
    
    Args:
        scheme: The URL scheme (http/https).
        netloc: The network location (domain).
        path: The URL path.
        state: The state parameter value.
    
    Returns:
        The constructed OAuth URL as a string.
    """
    # Ensure path starts with /
    if not path.startswith('/'):
        path = '/' + path
    
    # Build the base URL
    base_url = urllib.parse.urlunparse((
        scheme,
        netloc,
        path,
        '',
        '',
        ''
    ))
    
    # Add the state parameter
    params = {'state': state}
    return base_url + '?' + urllib.parse.urlencode(params)

def build_oauth_redirect_url(main_url, redirect_path, nonce):
    """
    Builds an OAuth callback URL for third-party authentication.
    
    Args:
        main_url: a string indicating the main URL of the application.
        redirect_path: a string specifying the path to execute OAuth callbacks.
        nonce: a string used to pass a state parameter for verifying the OAuth transaction.
    
    Returns:
        Returns the full OAuth callback URL as a string.
    
    Raises:
        ValueError: raised if the callback URL points to an invalid or unauthorized domain.
    """
    logger.info(f"Attempting to build OAuth redirect URL for main_url: {main_url}")
    
    # Parse the main URL to extract its components
    parsed_url = urllib.parse.urlparse(main_url)
    
    # Guard clause: validate domain
    if not validate_callback_domain(parsed_url.netloc):
        logger.error(f"Failed to build OAuth redirect URL: Invalid domain {parsed_url.netloc}")
        raise ValueError(f"Invalid or unauthorized domain: {parsed_url.netloc}")
    
    # Construct the OAuth URL
    oauth_url = construct_oauth_url(
        parsed_url.scheme,
        parsed_url.netloc,
        redirect_path,
        nonce
    )
    
    logger.info(f"Successfully built OAuth redirect URL: {oauth_url}")
    return oauth_url
