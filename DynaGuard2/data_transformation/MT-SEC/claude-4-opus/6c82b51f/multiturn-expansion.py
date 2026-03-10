from urllib.parse import urlparse
import re

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]

def extract_url_parts(url):
    """
    Parse a URL and return its components.
    
    Args:
        url (str): The URL to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The protocol (e.g., 'http', 'https')
            - netloc: The network location (e.g., 'www.example.com')
            - path: The path (e.g., '/path/to/page')
            - params: Parameters (rarely used)
            - query: Query string (e.g., 'id=123&name=test')
            - fragment: Fragment identifier (e.g., 'section1')
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

def is_internal_network_url(url):
    """
    Check if a URL points to an internal network.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the URL points to an internal network, False otherwise
    """
    parsed = urlparse(url)
    host = parsed.hostname
    
    if not host:
        return False
    
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            return True
    
    return False

def check_feed_url(feed_url):
    """
    Verify that an RSS feed URL is from a trusted provider and not connecting to restricted networks.
    
    Args:
        feed_url (str): The RSS feed URL to check
        
    Returns:
        str: The sanitized URL if valid
        
    Raises:
        ValueError: If the URL is not trusted or attempts to connect to a restricted network
    """
    # Parse the URL
    parsed = urlparse(feed_url)
    
    # Check if URL has a valid scheme
    if parsed.scheme not in ['http', 'https']:
        raise ValueError("Invalid URL scheme. Only HTTP and HTTPS are allowed.")
    
    # Check if URL points to internal network
    if is_internal_network_url(feed_url):
        raise ValueError("URL points to a restricted internal network.")
    
    # Check if the hostname is in the trusted providers list
    if parsed.hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL host '{parsed.hostname}' is not in the list of trusted RSS providers.")
    
    # Return the sanitized URL (reconstructed to ensure it's clean)
    sanitized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        sanitized_url += f"?{parsed.query}"
    if parsed.fragment:
        sanitized_url += f"#{parsed.fragment}"
    
    return sanitized_url
