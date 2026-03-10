from urllib.parse import urlparse

def extract_path(url):
    parsed_url = urlparse(url)
    return parsed_url.path
