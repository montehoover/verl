import re

def find_urls(text):
    # Pattern to match URLs (http, https, ftp, www)
    url_pattern = r'(?:(?:https?|ftp):\/\/|www\.)[^\s<>"{}|\\^`\[\]]+(?:\.[^\s<>"{}|\\^`\[\]]+)*'
    
    # Find all URL-like patterns in the text
    urls = re.findall(url_pattern, text)
    
    return urls
