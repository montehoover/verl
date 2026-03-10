from urllib.parse import urlparse

APPROVED_DOMAINS = ['example.com', 'myapp.com', 'trusted.com']

def is_valid_domain(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Handle www prefix
        if domain.startswith('www.'):
            domain = domain[4:]
            
        return domain in APPROVED_DOMAINS
    except:
        return False
