import re

def extract_components(text):
    # Pattern to match words, numbers, and symbols
    # \w+ matches word characters (letters, digits, underscore)
    # \d+\.?\d* matches integers and decimals
    # [^\w\s] matches symbols (non-word, non-space characters)
    pattern = r'\w+|\d+\.?\d*|[^\w\s]'
    
    components = re.findall(pattern, text)
    return components

def identify_url_scheme(url):
    # Pattern to match URL scheme (protocol)
    # ^([a-zA-Z][a-zA-Z0-9+.-]*): matches scheme at start of URL
    pattern = r'^([a-zA-Z][a-zA-Z0-9+.-]*):'
    
    match = re.match(pattern, url)
    if match:
        return match.group(1)
    else:
        return 'no_scheme'

def check_javascript_scheme(url_str):
    # Pattern to match harmful URL schemes
    # Case-insensitive match for javascript:, vbscript:, and other dangerous schemes
    harmful_pattern = r'^(javascript|vbscript|data):'
    
    # Pattern to match safe data URLs (image types)
    safe_data_pattern = r'^data:image/(jpeg|jpg|png|gif|bmp|webp|svg\+xml);base64,'
    
    # Check if it matches safe data URLs first
    if re.match(safe_data_pattern, url_str, re.IGNORECASE):
        return False
    
    # Check if it matches harmful patterns
    if re.match(harmful_pattern, url_str, re.IGNORECASE):
        return True
    
    return False
