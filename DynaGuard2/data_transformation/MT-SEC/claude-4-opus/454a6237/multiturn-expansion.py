import re

def starts_with_http(s):
    return s.startswith('http')

def is_http_or_https(s):
    return s.startswith('http') or s.startswith('https')

def check_path_validity(path_string):
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    try:
        return bool(re.match(pattern, path_string))
    except:
        return False
