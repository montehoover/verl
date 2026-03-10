import re

def find_urls(text):
    # Pattern to match URLs (http, https, ftp schemes and www)
    url_pattern = r'(?:(?:https?|ftp):\/\/|www\.)[^\s<>"{}|\\^\[\]`]+'
    
    # Find all matches
    urls = re.findall(url_pattern, text)
    
    return urls
