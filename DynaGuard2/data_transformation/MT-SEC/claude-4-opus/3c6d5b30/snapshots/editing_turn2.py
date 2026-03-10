from urllib.parse import urlparse, urljoin, urlencode

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

def construct_and_validate_url(base_url, path, query_params):
    # Construct the full URL with path
    full_url = urljoin(base_url, path)
    
    # Add query parameters if any
    if query_params:
        query_string = urlencode(query_params)
        # Check if URL already has query parameters
        if '?' in full_url:
            full_url += '&' + query_string
        else:
            full_url += '?' + query_string
    
    # Validate the domain
    if not is_valid_domain(full_url):
        raise ValueError(f"Domain not in allowed list: {urlparse(full_url).netloc}")
    
    return full_url
