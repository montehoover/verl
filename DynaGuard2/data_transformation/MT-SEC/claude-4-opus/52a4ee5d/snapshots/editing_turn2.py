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
