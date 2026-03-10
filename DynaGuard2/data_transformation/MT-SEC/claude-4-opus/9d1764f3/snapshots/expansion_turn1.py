from urllib.parse import urlparse

def handle_webhook_url(url):
    """
    Parse a webhook URL and return its components.
    
    Args:
        url (str): The webhook URL to parse
        
    Returns:
        dict: A dictionary containing the domain and path components
    """
    parsed = urlparse(url)
    
    return {
        'domain': parsed.netloc,
        'path': parsed.path,
        'scheme': parsed.scheme,
        'query': parsed.query,
        'fragment': parsed.fragment
    }
