from urllib.parse import urlparse
import ipaddress
import socket

def extract_url_parts(url):
    """
    Takes a URL string and breaks it down into its components.
    
    Args:
        url (str): The URL to parse
        
    Returns:
        dict: A dictionary containing the URL components
    """
    parsed = urlparse(url)
    
    return {
        'scheme': parsed.scheme,
        'hostname': parsed.hostname,
        'path': parsed.path,
        'port': parsed.port,
        'query': parsed.query,
        'fragment': parsed.fragment,
        'username': parsed.username,
        'password': parsed.password
    }

def validate_url(url):
    """
    Validates that a URL uses http/https and doesn't point to internal addresses.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if the URL is valid, False otherwise
    """
    try:
        parsed = urlparse(url)
        
        # Check if scheme is http or https
        if parsed.scheme not in ['http', 'https']:
            return False
        
        # Check if hostname exists
        if not parsed.hostname:
            return False
        
        # Check for localhost variations
        hostname_lower = parsed.hostname.lower()
        if hostname_lower in ['localhost', 'localhost.localdomain']:
            return False
        
        # Try to resolve hostname to IP address
        try:
            # Get IP address from hostname
            ip = socket.gethostbyname(parsed.hostname)
            ip_obj = ipaddress.ip_address(ip)
            
            # Check if it's a private IP address
            if ip_obj.is_private:
                return False
            
            # Check if it's a loopback address
            if ip_obj.is_loopback:
                return False
            
            # Check if it's a link local address
            if ip_obj.is_link_local:
                return False
            
        except (socket.gaierror, ValueError):
            # If hostname is already an IP, try to parse it directly
            try:
                ip_obj = ipaddress.ip_address(parsed.hostname)
                if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
                    return False
            except ValueError:
                # If it's not a valid IP and couldn't be resolved, consider it invalid
                return False
        
        return True
        
    except Exception:
        return False
