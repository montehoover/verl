from urllib.parse import urlparse

def validate_url(url):
    """
    Validates if a URL is properly formatted.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if the URL is valid
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url:
        raise ValueError("URL cannot be empty")
    
    try:
        result = urlparse(url)
        
        # Check if scheme and netloc are present
        if not result.scheme:
            raise ValueError("URL must have a scheme (e.g., http, https)")
        
        if not result.netloc:
            raise ValueError("URL must have a network location")
        
        # Check for valid schemes
        valid_schemes = ['http', 'https', 'ftp', 'ftps']
        if result.scheme not in valid_schemes:
            raise ValueError(f"URL scheme must be one of: {', '.join(valid_schemes)}")
        
        return True
        
    except Exception as e:
        raise ValueError(f"Invalid URL: {str(e)}")


def construct_resource_path(resource_id, resource_type):
    """
    Constructs a resource path based on resource type and identifier.
    
    Args:
        resource_id (str): The unique identifier for the resource
        resource_type (str): The type of resource (e.g., 'image', 'script', 'style')
        
    Returns:
        str: The constructed resource path
    """
    # Define resource type to directory mapping
    resource_directories = {
        'image': 'images',
        'script': 'scripts',
        'style': 'styles',
        'font': 'fonts',
        'video': 'videos',
        'audio': 'audio',
        'document': 'documents'
    }
    
    # Normalize resource type to lowercase
    resource_type_lower = resource_type.lower()
    
    # Get the directory for the resource type
    if resource_type_lower in resource_directories:
        directory = resource_directories[resource_type_lower]
    else:
        # Default to 'resources' for unknown types
        directory = 'resources'
    
    # Construct and return the path
    return f"/{directory}/{resource_id}"
