from urllib.parse import urlparse

def extract_domain(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    if not domain and not parsed.scheme:
        # Handle URLs without scheme (e.g., "example.com/path")
        parsed = urlparse('http://' + url)
        domain = parsed.netloc
    return domain

def is_trusted_domain(url):
    trusted_domains = [
        'google.com',
        'github.com',
        'stackoverflow.com',
        'wikipedia.org',
        'python.org'
    ]
    
    domain = extract_domain(url)
    
    # Remove www. prefix if present
    if domain.startswith('www.'):
        domain = domain[4:]
    
    return domain in trusted_domains
