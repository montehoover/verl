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

def verify_javascript_protocol(url_input):
    """
    Verify if a URL contains potentially harmful schemes like javascript:, vbscript:, etc.
    Allows safe data URLs for images.
    
    Args:
        url_input (str): The URL input to verify
        
    Returns:
        bool: True if potentially harmful, False if safe
    """
    # List of harmful URL schemes
    harmful_schemes = [
        'javascript:',
        'vbscript:',
        'livescript:',
        'mocha:',
        'jscript:',
        'view-source:',
        'about:',
        'chrome:',
        'chrome-extension:',
        'data:text/html',
        'data:application/javascript'
    ]
    
    # Convert to lowercase for case-insensitive matching
    url_lower = url_input.lower().strip()
    
    # Check for harmful schemes
    for scheme in harmful_schemes:
        if url_lower.startswith(scheme):
            return True
    
    # Check if it's a data URL that's not an image
    if url_lower.startswith('data:'):
        # Allow image data URLs
        image_pattern = r'^data:image/(jpeg|jpg|png|gif|webp|svg\+xml|bmp|ico);'
        if not re.match(image_pattern, url_lower):
            return True
    
    return False
