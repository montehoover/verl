from urllib.parse import urlparse, urljoin

def validate_url(url):
    """
    Validates a URL string.
    
    Args:
        url: A string representing the URL to validate
        
    Returns:
        True if the URL is valid
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not url:
        raise ValueError("URL cannot be empty")
    
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    try:
        parsed = urlparse(url)
        
        # Check if scheme is present and valid
        if not parsed.scheme:
            raise ValueError("URL must include a scheme (e.g., http://, https://)")
        
        # Check if scheme is allowed
        allowed_schemes = ['http', 'https', 'ftp', 'ftps']
        if parsed.scheme not in allowed_schemes:
            raise ValueError(f"URL scheme must be one of: {', '.join(allowed_schemes)}")
        
        # Check if netloc (domain) is present
        if not parsed.netloc:
            raise ValueError("URL must include a domain")
        
        # Basic domain validation
        if '..' in parsed.netloc:
            raise ValueError("Invalid domain format")
        
        # Check for invalid characters in domain
        if any(char in parsed.netloc for char in [' ', '<', '>', '"', '{', '}', '|', '\\', '^', '`']):
            raise ValueError("Domain contains invalid characters")
        
        return True
        
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid URL format: {str(e)}")


def build_url_with_path(base_url, path_component):
    """
    Combines a base URL with a path component to form a complete URL.
    
    Args:
        base_url: The base URL string
        path_component: The path component to append
        
    Returns:
        A properly formatted combined URL string
    """
    if not base_url:
        raise ValueError("Base URL cannot be empty")
    
    if not isinstance(base_url, str) or not isinstance(path_component, str):
        raise ValueError("Both base_url and path_component must be strings")
    
    # Ensure base_url ends with a slash for proper joining
    if base_url and not base_url.endswith('/'):
        base_url += '/'
    
    # Remove leading slash from path_component to avoid double slashes
    if path_component.startswith('/'):
        path_component = path_component[1:]
    
    # Use urljoin to properly combine the URLs
    combined_url = urljoin(base_url, path_component)
    
    return combined_url
