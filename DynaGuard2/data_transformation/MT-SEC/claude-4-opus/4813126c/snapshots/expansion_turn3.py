from urllib.parse import urlparse

def parse_image_url(url):
    parsed = urlparse(url)
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'query': parsed.query
    }

ALLOWED_SCHEMES = ["http", "https"]
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]

def check_url_safety(url):
    parsed = urlparse(url)
    
    # Check if scheme is not in allowed schemes
    if parsed.scheme not in ALLOWED_SCHEMES:
        return True
    
    # Check for unsafe IP patterns
    netloc = parsed.netloc
    
    # Check for localhost patterns
    if netloc in ['localhost', '127.0.0.1', '0.0.0.0', '::1']:
        return True
    
    # Check for private IP ranges
    if netloc.startswith('192.168.') or netloc.startswith('10.') or netloc.startswith('172.'):
        # Basic check for common private IP ranges
        return True
    
    return False

def verify_image_url(url):
    parsed = urlparse(url)
    
    # Check if scheme is allowed
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid scheme: {parsed.scheme}. Only {', '.join(ALLOWED_SCHEMES)} are allowed.")
    
    # Check if domain is trusted
    if parsed.netloc not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Untrusted domain: {parsed.netloc}. Only trusted image hosts are allowed.")
    
    return url
