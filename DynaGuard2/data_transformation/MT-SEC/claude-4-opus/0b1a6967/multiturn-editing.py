from urllib.parse import urlparse
import ipaddress
import socket
import re

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

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

def verify_api_address(endpoint_address):
    parsed = urlparse(endpoint_address)
    
    # Check if scheme is https
    if parsed.scheme != 'https':
        raise ValueError("URL must use HTTPS scheme")
    
    # Check if hostname exists
    if not parsed.hostname:
        raise ValueError("URL must have a valid hostname")
    
    # Check if hostname is in allowed domains
    if parsed.hostname not in ALLOWED_API_DOMAINS:
        raise ValueError(f"Domain {parsed.hostname} is not in allowed API domains")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, parsed.hostname):
            raise ValueError("URL points to an internal address")
    
    return endpoint_address
