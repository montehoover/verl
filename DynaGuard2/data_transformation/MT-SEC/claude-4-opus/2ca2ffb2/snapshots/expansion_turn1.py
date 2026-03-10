from urllib.parse import urlparse

def parse_image_url(url):
    """
    Parse an image URL and return its components.
    
    Args:
        url (str): The URL to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: URL scheme (e.g., 'http', 'https')
            - netloc: Network location (e.g., 'example.com')
            - path: URL path (e.g., '/images/photo.jpg')
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
