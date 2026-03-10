import re

def extract_components(text):
    # Extract words (sequences of alphabetic characters)
    words = re.findall(r'[a-zA-Z]+', text)
    
    # Extract numbers (integers and floats)
    numbers = re.findall(r'\d+\.?\d*', text)
    
    # Extract special characters (non-alphanumeric, non-whitespace)
    special_chars = re.findall(r'[^\w\s]', text)
    
    # Combine all components into a single list
    components = []
    components.extend(words)
    components.extend(numbers)
    components.extend(special_chars)
    
    return components

def identify_url_scheme(url):
    # Match URL scheme pattern (letters followed by ://)
    match = re.match(r'^([a-zA-Z]+)://', url)
    
    if match:
        return match.group(1)
    else:
        return 'unknown_scheme'

def detect_malicious_url_scheme(url_string):
    # List of malicious URL schemes
    malicious_schemes = [
        'javascript:',
        'vbscript:',
        'file:',
        'res:',
        'ms-its:',
        'mhtml:',
        'chrome:',
        'opera:',
        'about:'
    ]
    
    # Check if the URL starts with data:image/ (safe image data URLs)
    if re.match(r'^data:image/', url_string, re.IGNORECASE):
        return False
    
    # Check for malicious schemes
    for scheme in malicious_schemes:
        if re.search(r'^' + re.escape(scheme), url_string, re.IGNORECASE):
            return True
    
    return False
