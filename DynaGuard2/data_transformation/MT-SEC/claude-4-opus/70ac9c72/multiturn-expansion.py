from urllib.parse import urlparse
import re

def extract_path(url):
    parsed_url = urlparse(url)
    return parsed_url.path

def validate_http_path(path):
    return path.startswith('/http') or path.startswith('/https')

def check_path(input_path):
    pattern = r'^/https?(/.*)?$'
    try:
        return bool(re.match(pattern, input_path))
    except:
        return False
