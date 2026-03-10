import re

def find_urls(text):
    # Pattern to match URLs
    url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    
    # Find all URLs in the text
    urls = re.findall(url_pattern, text)
    
    return urls

def categorize_urls(text):
    # Pattern to match URLs with scheme capture group
    url_pattern = r'(https?)://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    
    # Find all URLs with their schemes
    matches = re.findall(url_pattern, text)
    
    # Categorize URLs by scheme
    categorized = {}
    for match in matches:
        scheme = match
        full_url = re.search(rf'{scheme}://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{{1,256}}\.[a-zA-Z0-9()]{{1,6}}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)', text).group()
        
        if scheme not in categorized:
            categorized[scheme] = []
        categorized[scheme].append(full_url)
    
    # Find URLs again to get full URLs properly
    url_pattern_full = r'(https?)://((?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*))'
    matches_full = re.findall(url_pattern_full, text)
    
    # Reorganize properly
    categorized = {}
    for scheme, rest in matches_full:
        full_url = f"{scheme}://{rest}"
        if scheme not in categorized:
            categorized[scheme] = []
        categorized[scheme].append(full_url)
    
    return categorized
