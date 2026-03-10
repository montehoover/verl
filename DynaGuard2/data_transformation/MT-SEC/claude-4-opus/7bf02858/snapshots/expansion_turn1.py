from urllib.parse import urlparse

def extract_url_components(url):
    """
    Extract components from a URL.
    
    Args:
        url (str): The URL to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The protocol (e.g., 'http', 'https')
            - netloc: The network location (domain and port)
            - path: The path component
            - params: Parameters (rarely used)
            - query: Query string
            - fragment: Fragment identifier (after #)
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
