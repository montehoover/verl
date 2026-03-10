from urllib.parse import urlparse, urljoin, urlencode

ALLOWED_DOMAINS = ['example.com', 'trusted.com', 'secure.org']

def construct_and_validate_url(base_url, path, query_params):
    # Construct the full URL
    full_url = urljoin(base_url, path)
    
    # Add query parameters if any
    if query_params:
        query_string = urlencode(query_params)
        full_url = f"{full_url}?{query_string}"
    
    # Parse and validate the domain
    try:
        parsed_url = urlparse(full_url)
        domain = parsed_url.netloc
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        
        if domain not in ALLOWED_DOMAINS:
            raise ValueError(f"Domain '{domain}' is not in the allowed list")
        
        return full_url
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid URL: {e}")
