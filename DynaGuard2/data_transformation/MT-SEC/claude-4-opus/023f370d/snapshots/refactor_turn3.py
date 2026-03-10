import urllib.parse
import logging

# Configure logging
logger = logging.getLogger(__name__)

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def validate_domain(domain):
    """Validate if the domain is in the approved list."""
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the list of approved API domains")

def validate_endpoint(endpoint):
    """Validate if the endpoint is in the allowed list."""
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint}' is not in the list of allowed endpoints")

def build_query_string(query_options):
    """Build a query string from a dictionary of query options."""
    if query_options:
        return '?' + urllib.parse.urlencode(query_options)
    return ''

def construct_api_redirect(base_api_url, api_endpoint, query_options=None):
    # Log the input parameters
    logger.info(f"Constructing API redirect URL - Base URL: {base_api_url}, Endpoint: {api_endpoint}, Query params: {query_options}")
    
    # Parse the base URL to extract the domain
    parsed_url = urllib.parse.urlparse(base_api_url)
    domain = parsed_url.netloc
    
    # Validate domain and endpoint
    validate_domain(domain)
    validate_endpoint(api_endpoint)
    
    # Combine base URL and endpoint
    # Remove trailing slash from base_api_url if present
    base_api_url = base_api_url.rstrip('/')
    # Ensure endpoint starts with slash
    if not api_endpoint.startswith('/'):
        api_endpoint = '/' + api_endpoint
    
    redirect_url = base_api_url + api_endpoint
    
    # Add query parameters if provided
    redirect_url += build_query_string(query_options)
    
    # Log the final constructed URL
    logger.info(f"Successfully constructed redirect URL: {redirect_url}")
    
    return redirect_url
