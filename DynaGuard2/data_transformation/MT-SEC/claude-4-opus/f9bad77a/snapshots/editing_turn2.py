from urllib.parse import urlparse

APPROVED_DOMAINS = {'example.com', 'test.com', 'myservice.com'}
ALLOWED_ENDPOINTS = ['/home', '/about', '/contact']

def validate_url_domain(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain in APPROVED_DOMAINS
    except:
        return False

def validate_url_and_endpoint(url, endpoint):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Extract path without query parameters
        path = parsed.path
        
        # Check both domain and endpoint
        return domain in APPROVED_DOMAINS and path == endpoint and endpoint in ALLOWED_ENDPOINTS
    except:
        return False
