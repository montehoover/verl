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

def validate_feed_url(url):
    """
    Ensures an RSS feed URL is from a trusted provider and not accessing internal networks.
    
    Args:
        url (str): The RSS feed URL to validate
        
    Returns:
        str: The validated feed URL if it meets all criteria
        
    Raises:
        ValueError: If the URL does not meet the security criteria
    """
    # Parse the URL
    parsed = urlparse(url)
    
    # Check if scheme is allowed
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme must be one of {ALLOWED_SCHEMES}")
    
    # Check if hostname exists
    if not parsed.hostname:
        raise ValueError("URL must have a valid hostname")
    
    # Check if hostname is from trusted provider
    if parsed.hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL must be from a trusted RSS provider: {TRUSTED_RSS_PROVIDERS}")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, parsed.hostname):
            raise ValueError("URL cannot point to internal network addresses")
    
    # Additional check: verify the hostname doesn't resolve to internal IPs
    try:
        # Try to resolve the hostname
        ips = socket.gethostbyname_ex(parsed.hostname)[2]
        for ip_str in ips:
            # Check if any resolved IP matches internal patterns
            for pattern in INTERNAL_IP_PATTERNS:
                if re.match(pattern, ip_str):
                    raise ValueError("URL resolves to internal network addresses")
            
            # Also check using ipaddress module
            try:
                ip = ipaddress.ip_address(ip_str)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    raise ValueError("URL resolves to internal network addresses")
            except ValueError:
                # Not a valid IP address format
                pass
    except (socket.gaierror, socket.herror):
        # Could not resolve hostname - this is acceptable for trusted providers
        pass
    
    return url
