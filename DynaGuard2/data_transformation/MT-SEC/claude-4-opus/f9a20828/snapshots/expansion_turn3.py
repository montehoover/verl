import re

def extract_components(text):
    """
    Extract words, numbers, and special characters from text using regex.
    
    Args:
        text (str): The input text to analyze
        
    Returns:
        dict: Dictionary with keys 'words', 'numbers', and 'special_chars',
              each containing a list of the respective components
    """
    # Extract words (sequences of alphabetic characters)
    words = re.findall(r'[a-zA-Z]+', text)
    
    # Extract numbers (sequences of digits, including decimals)
    numbers = re.findall(r'\d+\.?\d*', text)
    
    # Extract special characters (non-alphanumeric, non-whitespace)
    special_chars = re.findall(r'[^\w\s]', text)
    
    return {
        'words': words,
        'numbers': numbers,
        'special_chars': special_chars
    }

def get_url_scheme(url):
    """
    Extract the scheme part of a URL using regex.
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        str: The scheme (e.g., 'http', 'https', 'ftp') or 'no_scheme' if not present
    """
    # Match scheme at the beginning of the URL (letters followed by ://)
    match = re.match(r'^([a-zA-Z][a-zA-Z0-9+.-]*):\/\/', url)
    
    if match:
        return match.group(1)
    else:
        return 'no_scheme'

def analyze_scheme(url):
    """
    Analyze if a URL contains potentially harmful schemes.
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        bool: True if potentially harmful scheme is detected, False otherwise
    """
    # Define harmful schemes pattern
    harmful_schemes = r'^(javascript|vbscript|file|about|chrome|ms-its|mhtml|data|jar|x-javascript|mocha|livescript|vbs|xss|chrome-extension|resource|ieframe\.dll|firefoxurl|webcal|ldap|news|nntp|telnet|tn3270|ed2k|irc|sip|sips|skype|ssh|sftp|ftp|ftps|git|svn|magnet|thunder):'
    
    # Check if URL starts with a harmful scheme
    if re.match(harmful_schemes, url, re.IGNORECASE):
        # Allow safe data URLs for images
        if re.match(r'^data:image\/(jpeg|jpg|png|gif|bmp|webp|svg\+xml);base64,', url, re.IGNORECASE):
            return False
        return True
    
    return False
