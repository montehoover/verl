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


def construct_resource_path(res_id, res_type):
    """
    Constructs a path for a resource based on its type and identifier.
    
    Args:
        res_id (str): The resource identifier
        res_type (str): The type of resource
        
    Returns:
        str: The constructed resource path
    """
    return f"/{res_type}/{res_id}"
