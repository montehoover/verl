import re

def find_urls(text):
    # Pattern to match URL-like strings with capturing groups for scheme
    # Captures the scheme (http://, https://, ftp://) or www. as separate groups
    url_pattern = r'((?:https?|ftp)://)([^\s]+)|(www\.)([^\s]+)'
    
    # Find all matches in the text
    matches = re.findall(url_pattern, text)
    
    # Process matches to extract URLs and their schemes
    results = []
    for match in matches:
        if match[0]:  # Has explicit scheme (http://, https://, ftp://)
            scheme = match[0].rstrip('://')
            url = match[0] + match[1]
        else:  # www. pattern (no explicit scheme)
            scheme = 'www'
            url = match[2] + match[3]
        results.append((url, scheme))
    
    return results
