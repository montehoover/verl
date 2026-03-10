from urllib.parse import urlparse

def parse_url(url):
    parsed = urlparse(url)
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path
    }
