from urllib.parse import urlparse

def parse_webhook_url(webhook_url):
    """
    Parse a webhook URL and extract its basic components.
    
    Args:
        webhook_url (str): The webhook URL to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The URL scheme (e.g., 'http', 'https')
            - hostname: The hostname/domain
            - path: The path component of the URL
    """
    parsed = urlparse(webhook_url)
    
    return {
        'scheme': parsed.scheme,
        'hostname': parsed.hostname,
        'path': parsed.path
    }
