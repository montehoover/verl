from urllib.parse import urlparse
import ipaddress
import socket

def extract_url_parts(url):
    parsed = urlparse(url)
    return {
        'scheme': parsed.scheme,
        'hostname': parsed.hostname,
        'port': parsed.port,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment,
        'username': parsed.username,
        'password': parsed.password
    }

def check_url_security(url):
    parsed = urlparse(url)
    
    # Check if scheme is secure
    if parsed.scheme != 'https':
        return 'insecure'
    
    # Check if hostname exists
    if not parsed.hostname:
        return 'insecure'
    
    # Check for internal network addresses
    try:
        # Try to resolve hostname to IP
        ip = socket.gethostbyname(parsed.hostname)
        ip_obj = ipaddress.ip_address(ip)
        
        # Check if it's a private or reserved address
        if ip_obj.is_private or ip_obj.is_reserved or ip_obj.is_loopback or ip_obj.is_link_local:
            return 'insecure'
            
    except (socket.gaierror, ValueError):
        # If it's not a valid IP address, check if it's localhost
        if parsed.hostname.lower() in ['localhost', '127.0.0.1', '::1']:
            return 'insecure'
    
    return 'secure'
