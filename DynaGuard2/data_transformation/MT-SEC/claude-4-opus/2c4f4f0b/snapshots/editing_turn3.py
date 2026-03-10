import re
from urllib.parse import urlparse

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]

def extract_url_parts(url):
    """
    Extract components from a URL string.
    
    Args:
        url (str): The URL string to parse
        
    Returns:
        dict: Dictionary containing URL components
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

def is_valid_url(url):
    """
    Check if a URL is valid based on scheme and domain criteria.
    
    Args:
        url (str): The URL string to validate
        
    Returns:
        bool: True if URL has http/https scheme and non-localhost domain
    """
    parsed = urlparse(url)
    
    # Check if scheme is http or https
    if parsed.scheme not in ['http', 'https']:
        return False
    
    # Check if domain is localhost or loopback
    domain = parsed.netloc.lower()
    if not domain:
        return False
    
    # Remove port if present
    domain = domain.split(':')[0]
    
    # Check for localhost variations and loopback addresses
    invalid_domains = ['localhost', '127.0.0.1', '[::1]', '::1']
    if domain in invalid_domains:
        return False
    
    # Check for loopback IP range (127.0.0.0/8)
    if domain.startswith('127.'):
        parts = domain.split('.')
        if len(parts) == 4 and all(part.isdigit() for part in parts):
            return False
    
    return True

def check_rss_source(source_url):
    """
    Verify an RSS feed URL is from a trusted provider and not accessing internal networks.
    
    Args:
        source_url (str): The RSS feed URL to validate
        
    Returns:
        str: The URL if it is valid and trusted
        
    Raises:
        ValueError: If URL does not meet security criteria
    """
    parsed = urlparse(source_url)
    
    # Check if scheme is allowed
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid scheme: {parsed.scheme}. Only {', '.join(ALLOWED_SCHEMES)} are allowed.")
    
    # Extract hostname without port
    hostname = parsed.netloc.lower()
    if ':' in hostname:
        hostname = hostname.split(':')[0]
    
    if not hostname:
        raise ValueError("URL must have a valid hostname")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal network address '{hostname}' is not allowed")
    
    # Check if hostname is in trusted providers
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"RSS provider '{hostname}' is not in the list of trusted providers")
    
    return source_url
