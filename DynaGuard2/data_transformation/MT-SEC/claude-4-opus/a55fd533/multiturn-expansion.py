import re

def extract_components(text):
    """
    Extract components from text including words, numbers, and special characters.
    
    Args:
        text (str): The input text to process
        
    Returns:
        list: A list of extracted components
    """
    # Pattern to match words, numbers, and special characters
    # \w+ matches word characters (letters, digits, underscore)
    # \d+ matches one or more digits
    # [^\w\s] matches special characters (not word chars or whitespace)
    pattern = r'\w+|[^\w\s]'
    
    components = re.findall(pattern, text)
    
    return components

def identify_url_scheme(url):
    """
    Identify the scheme part of a URL.
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        str: The scheme (e.g., 'http', 'https', 'javascript') or 'no_scheme'
    """
    # Pattern to match URL scheme (characters before ://)
    pattern = r'^([a-zA-Z][a-zA-Z0-9+.-]*):(?://|:)'
    
    match = re.match(pattern, url)
    
    if match:
        return match.group(1).lower()
    else:
        return 'no_scheme'

def contains_dangerous_scheme(user_input):
    """
    Check if the user input contains dangerous URL schemes.
    
    Args:
        user_input (str): The input string to check
        
    Returns:
        bool: True if dangerous scheme found, False otherwise
    """
    # List of dangerous schemes
    dangerous_schemes = [
        'javascript', 'vbscript', 'livescript', 'mocha', 
        'jscript', 'view-source', 'about', 'ms-its',
        'mhtml', 'x-schema', 'res', 'ie.http'
    ]
    
    # Pattern to find URL schemes, excluding data:image URLs
    # Negative lookahead to exclude data:image patterns
    pattern = r'(?:^|[^a-zA-Z0-9])([a-zA-Z][a-zA-Z0-9+.-]*):(?!image/(?:jpeg|jpg|png|gif|bmp|webp|svg\+xml);base64)'
    
    matches = re.finditer(pattern, user_input, re.IGNORECASE)
    
    for match in matches:
        scheme = match.group(1).lower()
        if scheme in dangerous_schemes:
            return True
    
    return False
