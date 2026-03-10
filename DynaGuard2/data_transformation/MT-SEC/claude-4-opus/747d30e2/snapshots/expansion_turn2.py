import re

def find_urls(text):
    """
    Find all URL-like patterns in the given text.
    
    Args:
        text (str): The text to search for URLs
        
    Returns:
        list: A list of URL-like patterns found in the text
    """
    # Regex pattern to match URLs
    url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    
    # Find all matches
    urls = re.findall(url_pattern, text)
    
    return urls

def extract_url_scheme(url):
    """
    Extract the scheme part of a URL.
    
    Args:
        url (str): The URL to extract the scheme from
        
    Returns:
        str: The scheme (e.g., 'http', 'https', 'javascript') or 'no_scheme' if not present
    """
    # Regex pattern to match URL scheme
    scheme_pattern = r'^([a-zA-Z][a-zA-Z0-9+.-]*):(?://)?'
    
    match = re.match(scheme_pattern, url)
    
    if match:
        return match.group(1)
    else:
        return 'no_scheme'
