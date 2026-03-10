from urllib.parse import urlparse
import ipaddress
import socket
import re

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
        url (str): The URL to parse
        
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

def check_url_security(url):
    """
    Check if a URL uses a secure scheme and points to an external address.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the URL is secure and external, False otherwise
    """
    parsed = urlparse(url)
    
    # Check if scheme is http or https
    if parsed.scheme not in ['http', 'https']:
        return False
    
    # Extract hostname from netloc (remove port if present)
    hostname = parsed.netloc.split(':')[0]
    
    if not hostname:
        return False
    
    try:
        # Try to parse as IP address
        ip = ipaddress.ip_address(hostname)
        
        # Check if it's a private or reserved IP
        if ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_link_local:
            return False
            
    except ValueError:
        # Not an IP address, it's a domain name
        # Check for common internal hostnames
        internal_hosts = ['localhost', 'localhost.localdomain']
        if hostname.lower() in internal_hosts:
            return False
        
        # Try to resolve the domain and check the resulting IP
        try:
            resolved_ip = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(resolved_ip)
            if ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_link_local:
                return False
        except (socket.gaierror, OSError):
            # Could not resolve hostname, assume it's external
            pass
    
    return True

def clean_feed_link(rss_link):
    """
    Ensure RSS feed URL comes from a trusted provider and prevent access to internal networks.
    
    Args:
        rss_link (str): The RSS feed URL to sanitize
        
    Returns:
        str: The sanitized feed URL if valid
        
    Raises:
        ValueError: If the URL does not meet security criteria
    """
    parsed = urlparse(rss_link)
    
    # Check scheme
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme must be one of {ALLOWED_SCHEMES}")
    
    # Extract hostname
    hostname = parsed.netloc.split(':')[0]
    
    if not hostname:
        raise ValueError("Invalid URL: no hostname found")
    
    # Check if hostname is in trusted providers
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL must be from a trusted provider: {TRUSTED_RSS_PROVIDERS}")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError("URL points to internal network address")
    
    return rss_link
