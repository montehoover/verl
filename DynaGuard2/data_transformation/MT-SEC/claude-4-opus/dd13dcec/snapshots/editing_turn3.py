import re
from urllib.parse import urlparse

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

def is_trusted_domain(url, trusted_domains):
    """
    Check if a URL's hostname is in the list of trusted domains.
    
    Args:
        url (str): The URL string to check
        trusted_domains (list): List of trusted domain names
        
    Returns:
        bool: True if the hostname is in the trusted domains list, False otherwise
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    
    if hostname is None:
        return False
    
    # Check exact match or subdomain match
    for domain in trusted_domains:
        if hostname == domain or hostname.endswith('.' + domain):
            return True
    
    return False

def sanitize_feed_url(feed_url):
    """
    Sanitize and validate an RSS feed URL to ensure it's from trusted providers
    and doesn't expose internal network resources.
    
    Args:
        feed_url (str): The RSS feed URL to sanitize
        
    Returns:
        str: The sanitized and validated RSS feed URL
        
    Raises:
        ValueError: If the URL doesn't meet security criteria
    """
    # Parse the URL
    parsed = urlparse(feed_url)
    
    # Check if scheme is allowed
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme. Only {', '.join(ALLOWED_SCHEMES)} are allowed.")
    
    # Check if hostname exists
    if not parsed.hostname:
        raise ValueError("Invalid URL: no hostname found.")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, parsed.hostname):
            raise ValueError("Access to internal network addresses is not allowed.")
    
    # Check if hostname is in trusted RSS providers
    if not is_trusted_domain(feed_url, TRUSTED_RSS_PROVIDERS):
        raise ValueError(f"URL must be from a trusted RSS provider. Allowed providers: {', '.join(TRUSTED_RSS_PROVIDERS)}")
    
    # Return the original URL if all checks pass
    return feed_url
