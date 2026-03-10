from urllib.parse import urlparse
import ipaddress
import socket

def extract_url_parts(url):
    """
    Extract components from a URL string.
    
    Args:
        url (str): The URL string to parse
        
    Returns:
        dict: A dictionary containing the URL components
    """
    parsed = urlparse(url)
    
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment,
        'is_secure': is_secure_url(url)
    }

def is_secure_url(url):
    """
    Check if a URL is secure (uses http/https and doesn't point to internal addresses).
    
    Args:
        url (str): The URL string to check
        
    Returns:
        bool: True if the URL is secure, False otherwise
    """
    parsed = urlparse(url)
    
    # Check if scheme is http or https
    if parsed.scheme not in ['http', 'https']:
        return False
    
    # Extract hostname
    hostname = parsed.hostname
    if not hostname:
        return False
    
    # Check for localhost
    if hostname.lower() in ['localhost', 'localhost.localdomain']:
        return False
    
    # Try to resolve hostname to IP
    try:
        # Check if hostname is already an IP address
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        # Not an IP address, try to resolve hostname
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(hostname))
        except (socket.gaierror, ValueError):
            # Cannot resolve hostname, consider it potentially unsafe
            return False
    
    # Check if IP is private or loopback
    if ip.is_private or ip.is_loopback or ip.is_link_local:
        return False
    
    # Check for other special addresses
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return False
    
    return True
