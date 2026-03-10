import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def extract_domain(base_url):
    """Extract domain from a URL."""
    parsed_url = urllib.parse.urlparse(base_url)
    return parsed_url.netloc

def validate_domain(domain):
    """Validate if domain is in approved list."""
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the list of approved API domains")
    return domain

def validate_endpoint(endpoint):
    """Validate if endpoint is in allowed list."""
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint}' is not in the list of allowed endpoints")
    return endpoint

def clean_base_url(base_url):
    """Remove trailing slash from base URL."""
    return base_url.rstrip('/')

def normalize_endpoint(endpoint):
    """Ensure endpoint starts with forward slash."""
    return endpoint if endpoint.startswith('/') else '/' + endpoint

def construct_url(base_url, endpoint):
    """Combine base URL and endpoint."""
    clean_url = clean_base_url(base_url)
    normalized_endpoint = normalize_endpoint(endpoint)
    return clean_url + normalized_endpoint

def add_query_params(url, params):
    """Add query parameters to URL if provided."""
    if params:
        query_string = urllib.parse.urlencode(params)
        return url + '?' + query_string
    return url

def create_api_redirect_url(base_url_api, api_endpoint, opts=None):
    # Pipeline: Extract -> Validate -> Construct -> Add params
    domain = extract_domain(base_url_api)
    validate_domain(domain)
    validate_endpoint(api_endpoint)
    
    url = construct_url(base_url_api, api_endpoint)
    final_url = add_query_params(url, opts)
    
    return final_url
