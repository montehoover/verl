from urllib.parse import urlparse
import ipaddress
import socket

def extract_url_parts(url):
    """
    Takes a URL string and breaks it down into its components.
    
    Args:
        url (str): The URL string to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The protocol (e.g., 'http', 'https')
            - hostname: The domain name
            - path: The path after the domain
            - port: The port number (if specified)
            - query: The query string (if present)
            - fragment: The fragment identifier (if present)
    """
    parsed = urlparse(url)
    
    return {
        'scheme': parsed.scheme,
        'hostname': parsed.hostname,
        'path': parsed.path,
        'port': parsed.port,
        'query': parsed.query,
        'fragment': parsed.fragment
    }

def check_url_security(url):
    """
    Checks if a URL is secure based on its scheme and whether it points to internal addresses.
    
    Args:
        url (str): The URL string to check
        
    Returns:
        bool: True if the URL is secure (uses https and doesn't point to internal addresses),
              False otherwise
    """
    try:
        parsed = urlparse(url)
        
        # Check if scheme is https (secure)
        if parsed.scheme != 'https':
            return False
        
        # Check if hostname exists
        if not parsed.hostname:
            return False
        
        # Check if hostname is an IP address
        try:
            ip = ipaddress.ip_address(parsed.hostname)
            # Check if it's a private or loopback address
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
        except ValueError:
            # Not an IP address, it's a domain name
            # Try to resolve the domain to check if it points to internal addresses
            try:
                # Get all IP addresses for the hostname
                ips = socket.gethostbyname_ex(parsed.hostname)[2]
                for ip_str in ips:
                    ip = ipaddress.ip_address(ip_str)
                    if ip.is_private or ip.is_loopback or ip.is_link_local:
                        return False
            except (socket.gaierror, socket.herror):
                # Could not resolve hostname
                return False
        
        return True
        
    except Exception:
        return False
