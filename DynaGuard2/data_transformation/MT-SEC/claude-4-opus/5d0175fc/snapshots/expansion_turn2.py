from urllib.parse import urlparse

def extract_protocol(url):
    """
    Extract the protocol from a given URL string.
    
    Args:
        url (str): The URL string to parse
        
    Returns:
        str or None: The protocol (e.g., 'http', 'https') or None if not present
    """
    try:
        parsed = urlparse(url)
        return parsed.scheme if parsed.scheme else None
    except Exception:
        return None

def parse_url_components(url):
    """
    Parse a URL string and return its components.
    
    Args:
        url (str): The URL string to parse
        
    Returns:
        dict: A dictionary with keys 'protocol', 'domain', and 'path'
    """
    try:
        parsed = urlparse(url)
        
        # Extract protocol
        protocol = parsed.scheme if parsed.scheme else None
        
        # Extract domain (netloc)
        domain = parsed.netloc if parsed.netloc else None
        
        # Extract path
        path = parsed.path if parsed.path else ''
        
        return {
            'protocol': protocol,
            'domain': domain,
            'path': path
        }
    except Exception:
        return {
            'protocol': None,
            'domain': None,
            'path': ''
        }
