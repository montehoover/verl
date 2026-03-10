import re

def find_urls(text):
    # Pattern to match URL-like strings
    # Matches http://, https://, ftp://, or www. followed by domain-like patterns
    url_pattern = r'(?:https?://|ftp://|www\.)[^\s]+'
    
    # Find all matches in the text
    urls = re.findall(url_pattern, text)
    
    return urls
