from urllib.parse import urlparse

def get_url_components(url):
    """
    Parse an image URL and return its fundamental components.
    
    Args:
        url (str): The image URL to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The protocol (e.g., 'http', 'https')
            - netloc: The network location (domain and port)
            - path: The path to the resource
            - params: Parameters (semicolon-separated)
            - query: Query string (after the ?)
            - fragment: Fragment identifier (after the #)
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
