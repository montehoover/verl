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
