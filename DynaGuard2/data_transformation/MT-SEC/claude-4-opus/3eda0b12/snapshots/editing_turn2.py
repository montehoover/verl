from urllib.parse import urlparse, urlencode, urlunparse

APPROVED_DOMAINS = ['example.com', 'myapp.com', 'trusted.com']

def construct_and_validate_url(base_url, params):
    try:
        parsed = urlparse(base_url)
        domain = parsed.netloc
        
        # Handle www prefix
        if domain.startswith('www.'):
            domain = domain[4:]
            
        if domain not in APPROVED_DOMAINS:
            raise ValueError(f"Domain '{domain}' is not in the approved list")
        
        # Construct query string
        query_string = urlencode(params)
        
        # Build the complete URL with query parameters
        parsed = parsed._replace(query=query_string)
        complete_url = urlunparse(parsed)
        
        return complete_url
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid URL: {str(e)}")
