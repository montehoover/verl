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
