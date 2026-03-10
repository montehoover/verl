from urllib.parse import urlparse

def validate_url(url):
    """
    Validates if a URL is valid.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if the URL is valid
        
    Raises:
        ValueError: If the URL is invalid
    """
    try:
        result = urlparse(url)
        # Check if scheme and netloc are present (minimum requirements for a valid URL)
        if not all([result.scheme, result.netloc]):
            raise ValueError(f"Invalid URL: '{url}' - missing scheme or network location")
        
        # Check if scheme is http or https
        if result.scheme not in ['http', 'https', 'ftp', 'ftps']:
            raise ValueError(f"Invalid URL: '{url}' - unsupported scheme '{result.scheme}'")
        
        return True
    except Exception as e:
        raise ValueError(f"Invalid URL: '{url}' - {str(e)}")
