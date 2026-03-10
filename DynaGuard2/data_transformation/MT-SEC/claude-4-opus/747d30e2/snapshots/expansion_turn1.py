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
