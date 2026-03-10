import urllib.parse
import logging

# Configure logger
logger = logging.getLogger(__name__)

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def validate_domain(domain):
    """Validate if the domain is in the approved list."""
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not approved")

def validate_endpoint(path):
    """Validate if the endpoint is in the allowed list."""
    if path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{path}' is not allowed")

def construct_url(base_url, path, params=None):
    """Construct the full URL with optional query parameters."""
    full_url = urllib.parse.urljoin(base_url, path)
    
    if params:
        query_string = urllib.parse.urlencode(params)
        full_url = f"{full_url}?{query_string}"
    
    return full_url

def create_api_redirect_link(base_url, path, params=None):
    # Log the URL creation attempt
    logger.info(f"Creating API redirect URL - Base URL: {base_url}, Path: {path}, Params: {params}")
    
    # Parse the base URL to extract domain
    parsed_url = urllib.parse.urlparse(base_url)
    domain = parsed_url.netloc
    
    # Validate domain and endpoint
    validate_domain(domain)
    validate_endpoint(path)
    
    # Construct and return the full URL
    final_url = construct_url(base_url, path, params)
    
    # Log the final constructed URL
    logger.info(f"Successfully created API redirect URL: {final_url}")
    
    return final_url
