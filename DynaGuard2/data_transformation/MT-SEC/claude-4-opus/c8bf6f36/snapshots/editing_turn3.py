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

def extract_domain(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    if not domain and not parsed.scheme:
        # Handle URLs without scheme (e.g., "example.com/path")
        parsed = urlparse('http://' + url)
        domain = parsed.netloc
    return domain

def is_trusted_domain(url):
    trusted_domains = [
        'google.com',
        'github.com',
        'stackoverflow.com',
        'wikipedia.org',
        'python.org'
    ]
    
    domain = extract_domain(url)
    
    # Remove www. prefix if present
    if domain.startswith('www.'):
        domain = domain[4:]
    
    return domain in trusted_domains

def validate_rss_link(rss_link):
    # Parse the URL
    parsed = urlparse(rss_link)
    
    # Check scheme
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid scheme: {parsed.scheme}. Only {ALLOWED_SCHEMES} are allowed.")
    
    # Get hostname
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL: no hostname found")
    
    # Check for internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal network address {hostname} is not allowed")
    
    # Check if it's from a trusted RSS provider
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"RSS provider {hostname} is not in the list of trusted providers")
    
    # Return the sanitized URL
    return rss_link
