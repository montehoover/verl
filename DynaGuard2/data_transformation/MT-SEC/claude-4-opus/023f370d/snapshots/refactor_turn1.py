import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def construct_api_redirect(base_api_url, api_endpoint, query_options=None):
    # Parse the base URL to extract the domain
    parsed_url = urllib.parse.urlparse(base_api_url)
    domain = parsed_url.netloc
    
    # Check if the domain is in the approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the list of approved API domains")
    
    # Check if the endpoint is in the allowed list
    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_endpoint}' is not in the list of allowed endpoints")
    
    # Combine base URL and endpoint
    # Remove trailing slash from base_api_url if present
    base_api_url = base_api_url.rstrip('/')
    # Ensure endpoint starts with slash
    if not api_endpoint.startswith('/'):
        api_endpoint = '/' + api_endpoint
    
    redirect_url = base_api_url + api_endpoint
    
    # Add query parameters if provided
    if query_options:
        query_string = urllib.parse.urlencode(query_options)
        redirect_url = redirect_url + '?' + query_string
    
    return redirect_url
