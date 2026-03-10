import re

def extract_urls(text):
    # Regular expression pattern to match URLs
    url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    
    # Find all URLs in the text
    urls = re.findall(url_pattern, text)
    
    return urls
