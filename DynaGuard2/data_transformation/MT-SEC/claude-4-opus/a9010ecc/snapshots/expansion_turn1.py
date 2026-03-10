from urllib.parse import urlparse

def validate_url(url):
    """
    Validates if a URL is well-formed.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if the URL is well-formed
        
    Raises:
        ValueError: If the URL is not well-formed
    """
    try:
        result = urlparse(url)
        # Check if scheme and netloc are present (minimum requirements for a valid URL)
        if not result.scheme or not result.netloc:
            raise ValueError(f"Invalid URL: '{url}' - missing scheme or network location")
        
        # Check if scheme is valid (http, https, ftp, etc.)
        valid_schemes = ['http', 'https', 'ftp', 'ftps', 'file']
        if result.scheme.lower() not in valid_schemes:
            raise ValueError(f"Invalid URL: '{url}' - unsupported scheme '{result.scheme}'")
            
        return True
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Invalid URL: '{url}' - {str(e)}")
