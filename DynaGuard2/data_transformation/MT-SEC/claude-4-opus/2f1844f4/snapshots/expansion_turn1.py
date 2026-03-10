def is_http_protocol(url):
    """Check if a URL starts with 'http' or 'https'."""
    return url.startswith('http://') or url.startswith('https://')
