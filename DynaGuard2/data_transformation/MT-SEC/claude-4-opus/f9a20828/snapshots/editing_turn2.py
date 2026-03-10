import re

def extract_urls(text):
    # Regular expression pattern to match URLs
    url_pattern = r'(https?)://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    
    # Find all URLs in the text
    matches = re.finditer(url_pattern, text)
    
    # Extract URLs and their schemes
    result = []
    for match in matches:
        url = match.group(0)
        scheme = match.group(1)
        result.append((url, scheme))
    
    return result
