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

def validate_api_endpoint(api_url):
    """
    Validate that a URL points to an external trusted API endpoint.
    
    Args:
        api_url (str): The API URL to validate
        
    Returns:
        str: The validated URL
        
    Raises:
        ValueError: If the URL is not from an allowed domain or is internal
    """
    # Parse the URL
    parsed = urlparse(api_url)
    
    # Check scheme is https
    if parsed.scheme != 'https':
        raise ValueError("URL must use HTTPS scheme")
    
    # Get hostname
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL: no hostname found")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError("URL points to internal/private address")
    
    # Check if hostname is an IP address
    try:
        ip = ipaddress.ip_address(hostname)
        # If it's an IP, check if it's private/internal
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("URL points to internal/private IP address")
    except ValueError:
        # Not an IP address, it's a domain name
        # Check if it's in allowed domains
        if hostname not in ALLOWED_API_DOMAINS:
            raise ValueError(f"Domain '{hostname}' is not in allowed API domains")
    
    # Additional check: resolve domain and verify it's not internal
    try:
        resolved_ip = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(resolved_ip)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("Domain resolves to internal/private IP address")
    except socket.gaierror:
        raise ValueError("Could not resolve domain")
    
    return api_url
