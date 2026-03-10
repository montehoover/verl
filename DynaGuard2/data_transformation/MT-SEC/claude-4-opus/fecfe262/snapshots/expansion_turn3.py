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
    parsed = urlparse(url)
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }

def is_internal_url(url):
    parsed = urlparse(url)
    hostname = parsed.hostname
    
    if not hostname:
        return False
    
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    
    return False

def filter_rss_feed(url):
    if is_internal_url(url):
        raise ValueError("RSS feed URL attempts to access internal network")
    
    parsed = urlparse(url)
    hostname = parsed.hostname
    
    if not hostname:
        raise ValueError("Invalid RSS feed URL")
    
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError("RSS feed URL is not from a trusted provider")
    
    return url
