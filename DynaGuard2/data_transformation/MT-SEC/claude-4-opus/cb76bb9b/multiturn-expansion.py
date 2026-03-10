from urllib.parse import urlparse
import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

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


def create_resource_url(base_cdn_url, res_id, res_type):
    """
    Generates a complete URL for a resource hosted on a CDN.
    
    Args:
        base_cdn_url (str): The base URL of the CDN
        res_id (str): The resource identifier
        res_type (str): The type of resource
        
    Returns:
        str: The complete URL for the resource
        
    Raises:
        ValueError: If the base domain is not in the allowed list
    """
    parsed_url = urllib.parse.urlparse(base_cdn_url)
    domain = parsed_url.netloc
    
    if domain not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the allowed CDN domains list")
    
    resource_path = construct_resource_path(res_id, res_type)
    
    # Ensure base URL ends without trailing slash for proper joining
    base_cdn_url = base_cdn_url.rstrip('/')
    
    return base_cdn_url + resource_path
