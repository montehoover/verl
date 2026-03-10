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
