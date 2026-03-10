from urllib.parse import urlparse

def extract_domain(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    if not domain and not parsed.scheme:
        # Handle URLs without scheme (e.g., "example.com/path")
        parsed = urlparse('http://' + url)
        domain = parsed.netloc
    return domain
