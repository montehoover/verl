from urllib.parse import urlparse

APPROVED_DOMAINS = {'example.com', 'test.com', 'myservice.com'}
ALLOWED_ENDPOINTS = ['/home', '/about', '/contact']

def validate_url_domain(url):
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain in APPROVED_DOMAINS
    except Exception:
        return False

def validate_url_with_endpoint(url, endpoint):
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Check if domain is approved
        if domain not in APPROVED_DOMAINS:
            return False
        
        # Check if endpoint is allowed
        if endpoint not in ALLOWED_ENDPOINTS:
            return False
        
        return True
    except Exception:
        return False
