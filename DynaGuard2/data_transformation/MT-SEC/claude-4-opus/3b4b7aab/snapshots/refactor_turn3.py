import urllib.parse
import logging

# Configure logger
logger = logging.getLogger(__name__)

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def combine_url_parts(root_url, callback_uri):
    """
    Combines root URL and callback URI into a single URL.
    
    Args:
        root_url: The base URL
        callback_uri: The callback path
        
    Returns:
        Combined URL string
    """
    if root_url.endswith('/') and callback_uri.startswith('/'):
        return root_url[:-1] + callback_uri
    elif not root_url.endswith('/') and not callback_uri.startswith('/'):
        return root_url + '/' + callback_uri
    else:
        return root_url + callback_uri

def validate_domain(url, allowed_domains):
    """
    Validates that the URL's domain is in the allowed list.
    
    Args:
        url: The URL to validate
        allowed_domains: Set of allowed domains
        
    Returns:
        The domain if valid
        
    Raises:
        ValueError: If domain is not allowed
    """
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc
    
    if domain not in allowed_domains:
        error_msg = f"Domain '{domain}' is not in the list of allowed callback domains"
        logger.error(f"Domain validation failed: {error_msg}")
        raise ValueError(error_msg)
    
    logger.debug(f"Domain '{domain}' validated successfully")
    return domain

def append_state_parameter(url, state):
    """
    Appends state parameter to URL.
    
    Args:
        url: The base URL
        state: The state value to append
        
    Returns:
        URL with state parameter appended
    """
    separator = '&' if '?' in url else '?'
    return f"{url}{separator}state={urllib.parse.quote(state)}"

def construct_oauth_callback_uri(root_url, callback_uri, session_state):
    """
    Constructs an OAuth callback URL used during third-party authentication.
    
    Args:
        root_url: A string defining the application's main base URL.
        callback_uri: A string that represents the specific path for the OAuth callback endpoint.
        session_state: A string to include a unique value for ensuring the integrity of the OAuth request flow.
    
    Returns:
        A string representing the finalized OAuth callback URL.
    
    Raises:
        ValueError: Triggered if the computed callback URL's domain is not within the list of allowed domains.
    """
    # Log initial inputs
    logger.info(f"Constructing OAuth callback URL with root_url='{root_url}', callback_uri='{callback_uri}', session_state='{session_state}'")
    
    try:
        # Pipeline: combine -> validate -> append state
        full_url = combine_url_parts(root_url, callback_uri)
        logger.debug(f"Combined URL: '{full_url}'")
        
        validate_domain(full_url, ALLOWED_CALLBACK_DOMAINS)
        
        final_url = append_state_parameter(full_url, session_state)
        logger.info(f"Successfully constructed OAuth callback URL: '{final_url}'")
        
        return final_url
    
    except ValueError as e:
        logger.error(f"Failed to construct OAuth callback URL: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while constructing OAuth callback URL: {e}")
        raise
