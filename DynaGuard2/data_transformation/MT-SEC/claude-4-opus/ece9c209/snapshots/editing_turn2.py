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
        'hostname': parsed.hostname,
        'port': parsed.port,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment,
        'netloc': parsed.netloc,
        'username': parsed.username,
        'password': parsed.password
    }

def check_url_security(url):
    """
    Check if a URL is secure by verifying it uses HTTPS and doesn't point to internal addresses.
    
    Args:
        url (str): The URL string to check
        
    Returns:
        bool: True if the URL is secure, False otherwise
    """
    try:
        parsed = urlparse(url)
        
        # Check if scheme is https
        if parsed.scheme != 'https':
            return False
        
        # Get hostname
        hostname = parsed.hostname
        if not hostname:
            return False
        
        # Check if it's an IP address
        try:
            ip = ipaddress.ip_address(hostname)
            # Check for private/internal IP addresses
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        except ValueError:
            # Not an IP address, it's a domain name
            # Resolve the domain to check its IP
            try:
                resolved_ip = socket.gethostbyname(hostname)
                ip = ipaddress.ip_address(resolved_ip)
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                    return False
            except (socket.gaierror, ValueError):
                # Could not resolve or invalid
                return False
        
        return True
        
    except Exception:
        return False
