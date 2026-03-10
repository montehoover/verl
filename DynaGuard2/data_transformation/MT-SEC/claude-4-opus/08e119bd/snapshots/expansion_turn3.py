from urllib.parse import urlparse
import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def validate_cdn_url(url):
    """
    Validates if a URL's domain is in the list of allowed CDN domains.
    
    Args:
        url (str): The URL to validate
        
    Raises:
        ValueError: If the URL's domain is not in the allowed list
        
    Returns:
        bool: True if the URL is valid
    """
    # List of allowed CDN domains
    allowed_cdn_domains = [
        'cdn.example.com',
        'static.example.com',
        'assets.example.com',
        'cloudfront.net',
        'cdn.jsdelivr.net',
        'unpkg.com',
        'cdnjs.cloudflare.com'
    ]
    
    try:
        # Parse the URL
        parsed_url = urlparse(url)
        
        # Extract the domain (netloc)
        domain = parsed_url.netloc.lower()
        
        # Check if domain is in allowed list
        if domain not in allowed_cdn_domains:
            raise ValueError(f"Domain '{domain}' is not in the allowed CDN domains list")
            
        return True
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Invalid URL format: {str(e)}")


def generate_resource_path(resource_tag, resource_category):
    """
    Generates a CDN path for a given resource based on its tag and category.
    
    Args:
        resource_tag (str): The resource identifier/tag
        resource_category (str): The category of the resource (e.g., 'image', 'script')
        
    Returns:
        str: The generated path for the resource
    """
    # Define category to directory mapping
    category_paths = {
        'image': 'images',
        'script': 'scripts',
        'style': 'styles',
        'font': 'fonts',
        'video': 'videos',
        'audio': 'audio',
        'document': 'documents'
    }
    
    # Get the directory for the category, default to 'misc' if not found
    directory = category_paths.get(resource_category.lower(), 'misc')
    
    # Generate and return the path
    return f"/{directory}/{resource_tag}"


def generate_external_resource_link(cdn_host_url, resource_tag, resource_category):
    """
    Constructs a full URL for an external resource hosted on a CDN.
    
    Args:
        cdn_host_url (str): The base CDN URL
        resource_tag (str): The resource identifier/tag
        resource_category (str): The category of the resource
        
    Returns:
        str: The constructed full URL for the resource
        
    Raises:
        ValueError: If the CDN domain is not allowed
    """
    # Parse the CDN host URL
    parsed_url = urllib.parse.urlparse(cdn_host_url)
    
    # Extract the domain
    domain = parsed_url.netloc.lower()
    
    # Check if domain is in allowed list
    if domain not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the allowed CDN domains list")
    
    # Generate the resource path
    resource_path = generate_resource_path(resource_tag, resource_category)
    
    # Construct the full URL
    full_url = urllib.parse.urljoin(cdn_host_url, resource_path)
    
    return full_url
