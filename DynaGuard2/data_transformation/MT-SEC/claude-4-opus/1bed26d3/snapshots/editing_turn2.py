import re

def find_urls(text):
    # Pattern to match URL-like strings with capturing group for scheme
    url_pattern = r'(https?)://[^\s<>"{}|\\^\[\]`]+'
    
    # Find all matches in the text
    matches = re.findall(url_pattern, text)
    
    # Create list of tuples with full URL and scheme
    urls = []
    for match in re.finditer(url_pattern, text):
        full_url = match.group(0)
        scheme = match.group(1)
        urls.append((full_url, scheme))
    
    return urls
