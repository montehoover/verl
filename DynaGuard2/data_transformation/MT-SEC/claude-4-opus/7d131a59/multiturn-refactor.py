import urllib.parse
import logging

# Configure logger
logger = logging.getLogger(__name__)

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def _normalize_url_parts(base_url, callback_path):
    """
    Normalize the base URL and callback path for proper concatenation.
    
    Args:
        base_url (str): The base URL to normalize.
        callback_path (str): The callback path to normalize.
        
    Returns:
        tuple: A tuple containing (normalized_base_url, normalized_callback_path).
    """
    normalized_base = base_url.rstrip('/')
    normalized_path = callback_path if callback_path.startswith('/') else '/' + callback_path
    return normalized_base, normalized_path


def _validate_domain(domain):
    """
    Validate that the domain is in the allowed callback domains.
    
    Args:
        domain (str): The domain to validate.
        
    Raises:
        ValueError: If the domain is not in ALLOWED_CALLBACK_DOMAINS.
    """
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        error_msg = f"Domain '{domain}' is not allowed for OAuth callbacks"
        logger.error(f"Domain validation failed: {error_msg}")
        raise ValueError(error_msg)


def _merge_query_parameters(existing_query, new_params):
    """
    Merge existing query parameters with new parameters.
    
    Args:
        existing_query (str): The existing query string from the URL.
        new_params (dict): New parameters to add to the query.
        
    Returns:
        dict: A dictionary containing all merged parameters.
    """
    if not existing_query:
        return new_params
    
    # Parse existing parameters
    existing_params = urllib.parse.parse_qs(existing_query)
    
    # Convert lists to single values for existing params
    normalized_params = {
        k: v[0] if isinstance(v, list) and len(v) == 1 else v 
        for k, v in existing_params.items()
    }
    
    # Merge with new parameters
    normalized_params.update(new_params)
    return normalized_params


def _build_final_url(parsed_url, query_params):
    """
    Build the final URL from parsed components and query parameters.
    
    Args:
        parsed_url (ParseResult): The parsed URL components.
        query_params (dict): The query parameters to include.
        
    Returns:
        str: The complete URL with all components and parameters.
    """
    query_string = urllib.parse.urlencode(query_params)
    
    return urllib.parse.urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        query_string,
        parsed_url.fragment
    ))


def create_oauth_callback_url(base_url, callback_path, state):
    """
    Generates an OAuth callback URL for a third-party authentication process.
    
    Args:
        base_url: a string representing the base URL of the application.
        callback_path: a string representing the path for the OAuth callback.
        state: a string representing a unique state parameter for the OAuth flow.
    
    Returns:
        Returns a string that is the complete OAuth callback URL.
    
    Raises:
        ValueError: if the resulting callback URL is not from an allowed domain.
    """
    logger.info(f"Starting OAuth callback URL generation - Base URL: {base_url}, "
                f"Callback path: {callback_path}, State: {state}")
    
    try:
        # Normalize URL parts
        normalized_base, normalized_path = _normalize_url_parts(base_url, callback_path)
        
        # Combine base URL and callback path
        callback_url = normalized_base + normalized_path
        logger.debug(f"Initial callback URL constructed: {callback_url}")
        
        # Parse the URL to extract components
        parsed_url = urllib.parse.urlparse(callback_url)
        
        # Validate the domain
        _validate_domain(parsed_url.netloc)
        logger.debug(f"Domain validation passed for: {parsed_url.netloc}")
        
        # Prepare state parameter
        new_params = {'state': state}
        
        # Merge with any existing query parameters
        all_params = _merge_query_parameters(parsed_url.query, new_params)
        
        # Build and return the final URL
        final_url = _build_final_url(parsed_url, all_params)
        logger.info(f"Successfully generated OAuth callback URL: {final_url}")
        
        return final_url
        
    except ValueError as e:
        logger.error(f"Failed to generate OAuth callback URL - Base URL: {base_url}, "
                     f"Callback path: {callback_path}, Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during OAuth callback URL generation - "
                     f"Base URL: {base_url}, Callback path: {callback_path}, "
                     f"Error: {type(e).__name__}: {str(e)}")
        raise
