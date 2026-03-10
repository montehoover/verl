import re

def extract_components(text):
    # Pattern to match words, numbers, and symbols
    # \w+ matches word characters (letters, digits, underscore)
    # \d+\.?\d* matches integers and decimals
    # [^\w\s] matches symbols (non-word, non-space characters)
    pattern = r'\w+|\d+\.?\d*|[^\w\s]'
    
    components = re.findall(pattern, text)
    return components
