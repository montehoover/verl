import urllib.parse
import logging

# Configure logger
logger = logging.getLogger(__name__)

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def validate_domain(domain):
    """
    Validates if the domain is in the approved API domains.
    
    Args:
        domain: The domain to validate.
    
    Raises:
        ValueError: If the domain is not approved.
    """
    if domain not in APPROVED_API_DOMAINS:
        logger.error(f"Domain validation failed: '{domain}' is not in approved domains {APPROVED_API_DOMAINS}")
        raise ValueError(f"Domain '{domain}' is not in the approved API domains")
    logger.debug(f"Domain validation successful: '{domain}'")

def validate_endpoint(endpoint):
    """
    Validates if the endpoint is in the allowed endpoints.
    
    Args:
        endpoint: The endpoint to validate.
    
    Raises:
        ValueError: If the endpoint is not allowed.
    """
    if endpoint not in ALLOWED_ENDPOINTS:
        logger.error(f"Endpoint validation failed: '{endpoint}' is not in allowed endpoints {ALLOWED_ENDPOINTS}")
        raise ValueError(f"Endpoint '{endpoint}' is not in the allowed endpoints")
    logger.debug(f"Endpoint validation successful: '{endpoint}'")

def add_query_parameters(url, query_params):
    """
    Adds query parameters to a URL.
    
    Args:
        url: The base URL to add parameters to.
        query_params: Dictionary of query parameters to add.
    
    Returns:
        The URL with query parameters appended.
    """
    if not query_params:
        logger.debug("No query parameters to add")
        return url
    
    query_string = urllib.parse.urlencode(query_params)
    result = f"{url}?{query_string}"
    logger.debug(f"Added query parameters: {query_params} -> {result}")
    return result

def build_api_redirect_url(api_base_url, endpoint, query_params=None):
    """
    Constructs a redirect URL for API responses.
    
    Args:
        api_base_url: A string representing the base URL of the API.
        endpoint: A string representing the specific API endpoint.
        query_params: An optional dictionary of query parameters.
    
    Returns:
        A string that is the complete API redirect URL.
    
    Raises:
        ValueError: If the resulting URL is not pointing to an approved API domain or endpoint.
    """
    logger.info(f"Building API redirect URL - base: '{api_base_url}', endpoint: '{endpoint}', params: {query_params}")
    
    try:
        # Parse the base URL to extract the domain
        parsed_base = urllib.parse.urlparse(api_base_url)
        domain = parsed_base.netloc
        logger.debug(f"Extracted domain: '{domain}'")
        
        # Validate domain and endpoint
        validate_domain(domain)
        validate_endpoint(endpoint)
        
        # Construct the full URL
        full_url = urllib.parse.urljoin(api_base_url, endpoint)
        logger.debug(f"Constructed base URL: '{full_url}'")
        
        # Add query parameters if provided
        final_url = add_query_parameters(full_url, query_params)
        
        logger.info(f"Successfully built API redirect URL: '{final_url}'")
        return final_url
        
    except ValueError as e:
        logger.error(f"Failed to build API redirect URL: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error building API redirect URL: {e}")
        raise
