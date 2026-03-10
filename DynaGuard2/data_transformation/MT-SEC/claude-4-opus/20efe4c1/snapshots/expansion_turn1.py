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
