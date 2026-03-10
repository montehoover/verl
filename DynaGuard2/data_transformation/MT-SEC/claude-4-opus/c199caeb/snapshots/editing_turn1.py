from urllib.parse import urlparse

APPROVED_DOMAINS = ['example.com', 'trusted.com']

def is_valid_domain(url):
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Handle cases where urlparse might not detect the scheme
        if not domain and '://' not in url:
            parsed_url = urlparse('http://' + url)
            domain = parsed_url.netloc
            
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        return domain in APPROVED_DOMAINS
    except Exception:
        return False
