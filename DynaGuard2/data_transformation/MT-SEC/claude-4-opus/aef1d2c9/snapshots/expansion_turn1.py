import re

def extract_components(text):
    # Define regex patterns for different components
    word_pattern = r'[a-zA-Z]+'
    number_pattern = r'\d+'
    special_char_pattern = r'[^a-zA-Z0-9\s]'
    
    # Find all matches for each pattern
    words = re.findall(word_pattern, text)
    numbers = re.findall(number_pattern, text)
    special_chars = re.findall(special_char_pattern, text)
    
    # Combine all components into a single list
    components = []
    
    # Use regex to find all components in order
    pattern = r'([a-zA-Z]+|\d+|[^a-zA-Z0-9\s])'
    components = re.findall(pattern, text)
    
    return components
