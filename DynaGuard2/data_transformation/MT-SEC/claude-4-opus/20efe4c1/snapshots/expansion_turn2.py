import re

def extract_components(text):
    """
    Extract and return a list of components from text including words, numbers, and special characters.
    
    Args:
        text (str): The input text to analyze
        
    Returns:
        list: A list of extracted components (words, numbers, special characters)
    """
    # Define regex patterns
    word_pattern = r'[a-zA-Z]+'
    number_pattern = r'\d+'
    special_char_pattern = r'[^\w\s]'
    
    # Extract components
    words = re.findall(word_pattern, text)
    numbers = re.findall(number_pattern, text)
    special_chars = re.findall(special_char_pattern, text)
    
    # Combine all components
    components = []
    
    # Use regex to find all components in order
    pattern = r'([a-zA-Z]+|\d+|[^\w\s])'
    components = re.findall(pattern, text)
    
    return components

def identify_url_scheme(url):
    """
    Extract the scheme part of a URL using regex.
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        str: The scheme (e.g., 'http', 'https', 'javascript') or 'no_scheme' if none found
    """
    # Regex pattern to match URL scheme
    pattern = r'^([a-zA-Z][a-zA-Z0-9+.-]*):\/\/'
    match = re.match(pattern, url)
    
    if match:
        return match.group(1)
    else:
        return 'no_scheme'
