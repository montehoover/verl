import re

def find_urls(text):
    # Pattern to match URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    
    # Find all matches
    urls = re.findall(url_pattern, text)
    
    return urls
