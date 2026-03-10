from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def extract_url_parts(url):
    """
    Parse a URL and extract its components.
    
    Args:
        url (str): The URL string to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The protocol (e.g., 'http', 'https')
            - netloc: The network location (domain and port if specified)
            - path: The path component of the URL
            - params: Parameters for the path element
            - query: Query string
            - fragment: Fragment identifier
    """
    parsed = urlparse(url)
    
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }

def check_internal_access(url):
    """
    Check if a URL targets local or private networks.
    
    Args:
        url (str): The URL string to check
        
    Returns:
        bool: True if the URL targets internal IP ranges, None otherwise
    """
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    
    # Extract hostname from netloc (remove port if present)
    hostname = netloc.split(':')[0]
    
    # Check for localhost variations
    if hostname in ['localhost', '127.0.0.1', '::1']:
        return True
    
    # Check for private IP ranges
    if hostname.startswith('10.') or hostname.startswith('192.168.'):
        return True
    
    # Check for other private ranges
    if hostname.startswith('172.'):
        # Check if it's in the 172.16.0.0 - 172.31.255.255 range
        parts = hostname.split('.')
        if len(parts) >= 2:
            try:
                second_octet = int(parts[1])
                if 16 <= second_octet <= 31:
                    return True
            except ValueError:
                pass
    
    return None

def validate_callback(url):
    """
    Validate a callback URL against approved domains and internal network restrictions.
    
    Args:
        url (str): The callback URL to validate
        
    Returns:
        str: The processed URL for safe internal API use
        
    Raises:
        ValueError: If the URL is not from an approved domain or targets internal networks
    """
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    
    # Extract hostname from netloc (remove port if present)
    hostname = netloc.split(':')[0]
    
    # Check if domain is in approved list
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"URL domain '{hostname}' is not in the approved domains list")
    
    # Check for internal network access
    for internal_range in INTERNAL_IP_RANGES:
        if hostname.startswith(internal_range) or hostname == internal_range:
            raise ValueError(f"URL targets internal network: {hostname}")
    
    # Additional check using the existing function
    if check_internal_access(url):
        raise ValueError(f"URL targets internal network: {hostname}")
    
    # Return the original URL if all checks pass
    return url
