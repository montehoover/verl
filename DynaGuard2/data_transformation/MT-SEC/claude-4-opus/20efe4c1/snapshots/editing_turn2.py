import re

def find_urls(text):
    # Pattern to match URLs with capturing groups for scheme
    url_pattern = r'((?:https?|ftp)|(?:www))(?::\/\/|\.)[^\s<>"{}|\\^\[\]`]+'
    
    # Find all matches with their full match
    matches = re.finditer(url_pattern, text)
    
    # Extract URLs and their schemes
    result = []
    for match in matches:
        full_url = match.group(0)
        scheme = match.group(1)
        
        # Handle www URLs which don't have a protocol scheme
        if scheme == 'www':
            scheme = 'http'  # Default to http for www URLs
            full_url = 'http://' + full_url
        
        result.append((full_url, scheme))
    
    return result
