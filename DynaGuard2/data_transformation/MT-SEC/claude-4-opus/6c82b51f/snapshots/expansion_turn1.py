from urllib.parse import urlparse

def extract_url_parts(url):
    """
    Parse a URL and return its components.
    
    Args:
        url (str): The URL to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The protocol (e.g., 'http', 'https')
            - netloc: The network location (e.g., 'www.example.com')
            - path: The path (e.g., '/path/to/page')
            - params: Parameters (rarely used)
            - query: Query string (e.g., 'id=123&name=test')
            - fragment: Fragment identifier (e.g., 'section1')
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
