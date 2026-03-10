import re

def find_urls(text):
    # Pattern to match URL-like strings
    url_pattern = r'https?://[^\s<>"{}|\\^\[\]`]+|www\.[^\s<>"{}|\\^\[\]`]+\.[^\s<>"{}|\\^\[\]`]+'
    
    # Find all matches in the text
    urls = re.findall(url_pattern, text)
    
    return urls
