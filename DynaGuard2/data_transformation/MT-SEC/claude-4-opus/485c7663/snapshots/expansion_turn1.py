from urllib.parse import urlparse

def parse_url(url):
    """
    Parse a URL and return its components.
    
    Args:
        url (str): The URL to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The protocol (e.g., 'http', 'https')
            - netloc: The network location (domain and port)
            - path: The path component
            - params: Parameters (semicolon-separated)
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
