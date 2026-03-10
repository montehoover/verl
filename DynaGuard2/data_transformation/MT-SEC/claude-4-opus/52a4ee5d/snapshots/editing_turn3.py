import re

def find_urls(text):
    # Pattern to match URLs with capturing groups for scheme
    url_pattern = r'((?:https?|ftp)|(?:www))(?::\/\/|\.)[^\s<>"{}|\\^`\[\]]+(?:\.[^\s<>"{}|\\^`\[\]]+)*'
    
    # Find all URL-like patterns in the text with their schemes
    matches = re.finditer(url_pattern, text)
    
    urls = []
    for match in matches:
        full_url = match.group(0)
        scheme = match.group(1)
        
        # If scheme is 'www', prepend it to make full URL and set scheme as 'http'
        if scheme == 'www':
            full_url = 'www.' + full_url[3:]
            scheme = 'http'
        
        urls.append((full_url, scheme))
    
    return urls

def detect_js_scheme(url_string):
    # Pattern to match dangerous URL schemes
    dangerous_schemes = r'^\s*(?:javascript|vbscript|data:(?!image\/)).*'
    
    try:
        # Check if the URL contains dangerous schemes
        if re.match(dangerous_schemes, url_string, re.IGNORECASE):
            return True
        return False
    except re.error:
        # If regex error occurs, return False (assume safe)
        return False
