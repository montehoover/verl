from urllib.parse import urlparse

def extract_url_components(url):
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment
