import re

def find_urls(text):
    # Pattern to match URL-like strings with optional scheme
    url_pattern = r'(https?://)([^\s<>"{}|\\^\[\]`]+)|(www\.[^\s<>"{}|\\^\[\]`]+\.[^\s<>"{}|\\^\[\]`]+)'
    
    # Find all matches in the text
    matches = re.findall(url_pattern, text)
    
    # Process matches to create (url, scheme) tuples
    results = []
    for match in matches:
        if match[0]:  # Has scheme (http:// or https://)
            url = match[0] + match[1]
            scheme = match[0].rstrip('://')
            results.append((url, scheme))
        elif match[2]:  # www. without scheme
            url = match[2]
            scheme = None  # No explicit scheme
            results.append((url, scheme))
    
    return results
