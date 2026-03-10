def get_domain_from_url(url):
    # Remove protocol if present
    if '://' in url:
        url = url.split('://', 1)[1]
    
    # Remove path if present
    if '/' in url:
        url = url.split('/', 1)[0]
    
    # Remove port if present
    if ':' in url:
        url = url.split(':', 1)[0]
    
    return url
