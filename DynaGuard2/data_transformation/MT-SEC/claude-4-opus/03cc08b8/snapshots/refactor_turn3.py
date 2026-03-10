import urllib.parse
import logging

# Configure logger
logger = logging.getLogger(__name__)

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def validate_callback_domain(callback_domain, app_domain, allowed_domains):
    """Validate that the callback domain is either the same as app domain or in allowed list."""
    logger.debug(f"Validating callback domain: {callback_domain} against app domain: {app_domain}")
    
    if callback_domain != app_domain and callback_domain not in allowed_domains:
        logger.error(f"Callback domain validation failed: '{callback_domain}' is not allowed")
        raise ValueError(f"Callback domain '{callback_domain}' is not allowed")
    
    logger.debug(f"Callback domain validation passed for: {callback_domain}")
    return True

def construct_callback_url(application_url, callback_route):
    """Construct the full callback URL from application URL and callback route."""
    logger.debug(f"Constructing callback URL from application_url: {application_url}, callback_route: {callback_route}")
    
    parsed_callback = urllib.parse.urlparse(callback_route)
    
    if parsed_callback.netloc:
        # callback_route contains a domain, validate it
        logger.debug(f"Callback route contains domain: {parsed_callback.netloc}")
        parsed_app = urllib.parse.urlparse(application_url)
        validate_callback_domain(parsed_callback.netloc, parsed_app.netloc, ALLOWED_CALLBACK_DOMAINS)
        result_url = callback_route
    else:
        # callback_route is just a path, combine with application URL
        logger.debug("Callback route is a path, joining with application URL")
        result_url = urllib.parse.urljoin(application_url, callback_route)
    
    logger.debug(f"Constructed callback URL: {result_url}")
    return result_url

def add_state_parameter(url, state_value):
    """Add or update the state parameter in the URL query string."""
    logger.debug(f"Adding state parameter to URL: {url}, state: {state_value}")
    
    parsed_url = urllib.parse.urlparse(url)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    query_params['state'] = [state_value]
    
    new_query = urllib.parse.urlencode(query_params, doseq=True)
    final_url = urllib.parse.urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        new_query,
        parsed_url.fragment
    ))
    
    logger.debug(f"URL with state parameter: {final_url}")
    return final_url

def assemble_oauth_callback(application_url, callback_route, token_state):
    """Assemble a complete OAuth callback URL with state parameter."""
    logger.info(f"Starting OAuth callback assembly - app_url: {application_url}, callback: {callback_route}, state: {token_state}")
    
    try:
        # Pipeline: construct URL -> add state parameter
        callback_url = construct_callback_url(application_url, callback_route)
        final_url = add_state_parameter(callback_url, token_state)
        
        logger.info(f"Successfully assembled OAuth callback URL: {final_url}")
        return final_url
    except Exception as e:
        logger.error(f"Failed to assemble OAuth callback URL: {str(e)}")
        raise
