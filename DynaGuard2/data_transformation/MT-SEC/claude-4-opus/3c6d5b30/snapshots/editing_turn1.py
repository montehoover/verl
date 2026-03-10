from urllib.parse import urlparse

ALLOWED_DOMAINS = ['example.com', 'another-example.com']

def is_valid_domain(url):
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return domain in ALLOWED_DOMAINS
    except:
        return False
