import urllib.parse
import logging

# Configure logger
logger = logging.getLogger(__name__)

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def validate_redirect_domain(redirect_url):
    """
    Validates that the redirect URL's domain is in the allowed list.
    
    Args:
        redirect_url: The URL to validate
        
    Returns:
        The domain if valid
        
    Raises:
        ValueError: If domain is not in the allowed list
    """
    logger.debug(f"Validating redirect URL: {redirect_url}")
    
    parsed_url = urllib.parse.urlparse(redirect_url)
    domain = parsed_url.netloc
    
    logger.debug(f"Extracted domain: {domain}")
    
    if domain not in ALLOWED_RETURN_DOMAINS:
        logger.error(f"Domain validation failed: '{domain}' is not in the approved domains list")
        raise ValueError(f"Domain '{domain}' is not in the approved domains list")
    
    logger.info(f"Domain validation successful: {domain}")
    return domain

def construct_sso_url(base_path, query_params):
    """
    Constructs a URL by appending query parameters to a base path.
    
    Args:
        base_path: The base SSO path
        query_params: Dictionary of query parameters
        
    Returns:
        The complete URL with query parameters
    """
    logger.debug(f"Constructing SSO URL with base path: {base_path}")
    logger.debug(f"Query parameters: {query_params}")
    
    query_string = urllib.parse.urlencode(query_params)
    logger.debug(f"Encoded query string: {query_string}")
    
    if '?' in base_path:
        sso_url = f"{base_path}&{query_string}"
        logger.debug("Base path already contains query parameters, appending with '&'")
    else:
        sso_url = f"{base_path}?{query_string}"
        logger.debug("Base path has no query parameters, appending with '?'")
    
    logger.debug(f"Constructed SSO URL: {sso_url}")
    return sso_url

def build_sso_url(sso_path, uid, redirect_url):
    logger.info(f"Building SSO URL for user: {uid}")
    logger.debug(f"SSO path: {sso_path}")
    logger.debug(f"Redirect URL: {redirect_url}")
    
    try:
        # Validate the redirect URL domain
        validate_redirect_domain(redirect_url)
        
        # Build the query parameters
        params = {
            'uid': uid,
            'redirect': redirect_url
        }
        
        # Construct and return the SSO URL
        sso_url = construct_sso_url(sso_path, params)
        
        logger.info(f"Successfully built SSO URL for user {uid}")
        return sso_url
        
    except ValueError as e:
        logger.error(f"Failed to build SSO URL for user {uid}: {str(e)}")
        raise
