from urllib.parse import urlparse

def extract_url_parts(url):
    """
    Takes a URL string and breaks it down into its components.
    
    Args:
        url (str): The URL string to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The protocol (e.g., 'http', 'https')
            - hostname: The domain name
            - path: The path after the domain
            - port: The port number (if specified)
            - query: The query string (if present)
            - fragment: The fragment identifier (if present)
    """
    parsed = urlparse(url)
    
    return {
        'scheme': parsed.scheme,
        'hostname': parsed.hostname,
        'path': parsed.path,
        'port': parsed.port,
        'query': parsed.query,
        'fragment': parsed.fragment
    }
