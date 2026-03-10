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

def filter_rss_feed(rss_feed_url):
    """
    Takes an RSS feed URL and ensures it is from a trusted provider.
    
    Args:
        rss_feed_url (str): The RSS feed URL to validate
        
    Returns:
        str: The sanitized feed URL if it is from an approved provider
        
    Raises:
        ValueError: If the URL does not meet the security criteria
    """
    parsed = urlparse(rss_feed_url)
    
    # Check if scheme is allowed
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme must be one of {ALLOWED_SCHEMES}")
    
    # Check if hostname exists
    if not parsed.hostname:
        raise ValueError("URL must have a valid hostname")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, parsed.hostname):
            raise ValueError("URL points to an internal network address")
    
    # Check if hostname is in trusted providers
    if parsed.hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"RSS feed must be from one of the trusted providers: {TRUSTED_RSS_PROVIDERS}")
    
    # Return the sanitized URL
    return rss_feed_url
