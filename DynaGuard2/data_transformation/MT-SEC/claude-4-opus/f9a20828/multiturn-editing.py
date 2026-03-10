import re

def extract_urls(text):
    # Regular expression pattern to match URLs
    url_pattern = r'(https?)://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    
    # Find all URLs in the text
    matches = re.finditer(url_pattern, text)
    
    # Extract URLs and their schemes
    result = []
    for match in matches:
        url = match.group(0)
        scheme = match.group(1)
        result.append((url, scheme))
    
    return result

def analyze_scheme(url):
    # Pattern to match harmful URL schemes
    harmful_schemes_pattern = r'^(javascript|vbscript|file|about|chrome|ms-\w+):'
    
    # Pattern to match legitimate image data URLs
    image_data_pattern = r'^data:image/(png|jpg|jpeg|gif|bmp|svg\+xml|webp);'
    
    # Check if it's a legitimate image data URL
    if re.match(image_data_pattern, url, re.IGNORECASE):
        return False
    
    # Check if it contains a harmful scheme
    if re.match(harmful_schemes_pattern, url, re.IGNORECASE):
        return True
    
    return False
