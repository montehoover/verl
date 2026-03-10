from urllib.parse import urlparse

def extract_path(url):
    parsed_url = urlparse(url)
    return parsed_url.path

def validate_http_path(path):
    return path.startswith('/http') or path.startswith('/https')
