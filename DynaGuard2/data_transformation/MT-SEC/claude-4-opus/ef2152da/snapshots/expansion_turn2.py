from urllib.parse import urlparse

def extract_url_parts(url):
    """
    Parse a URL and extract its components.
    
    Args:
        url (str): The URL string to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The protocol (e.g., 'http', 'https')
            - netloc: The network location (domain and port if specified)
            - path: The path component of the URL
            - params: Parameters for the path element
            - query: Query string
            - fragment: Fragment identifier
    """
    parsed = urlparse(url)
    
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }

def check_internal_access(url):
    """
    Check if a URL targets local or private networks.
    
    Args:
        url (str): The URL string to check
        
    Returns:
        bool: True if the URL targets internal IP ranges, None otherwise
    """
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    
    # Extract hostname from netloc (remove port if present)
    hostname = netloc.split(':')[0]
    
    # Check for localhost variations
    if hostname in ['localhost', '127.0.0.1', '::1']:
        return True
    
    # Check for private IP ranges
    if hostname.startswith('10.') or hostname.startswith('192.168.'):
        return True
    
    # Check for other private ranges
    if hostname.startswith('172.'):
        # Check if it's in the 172.16.0.0 - 172.31.255.255 range
        parts = hostname.split('.')
        if len(parts) >= 2:
            try:
                second_octet = int(parts[1])
                if 16 <= second_octet <= 31:
                    return True
            except ValueError:
                pass
    
    return None
