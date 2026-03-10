import re

def find_urls(text):
    # Pattern to match URLs with capturing group for scheme
    url_pattern = r'(https?)://[^\s<>"{}|\\^`\[\]]+'
    
    # Find all matches
    matches = re.finditer(url_pattern, text)
    
    # Extract URLs and their schemes
    result = []
    for match in matches:
        url = match.group(0)
        scheme = match.group(1)
        result.append((url, scheme))
    
    return result
