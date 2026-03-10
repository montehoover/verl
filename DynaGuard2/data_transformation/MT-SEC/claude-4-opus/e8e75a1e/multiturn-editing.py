import re

def find_urls(text):
    # Pattern to match URLs with capturing group for scheme
    url_pattern = r'(https?)://[^\s<>"{}|\\^`\[\]]+'
    
    # Find all matches
    matches = re.finditer(url_pattern, text)
    
    # Extract URLs and their schemes
    result = []
    for match in matches:
        url = match.group(0)
        scheme = match.group(1)
        result.append((url, scheme))
    
    return result

def detect_malicious_url_scheme(url_string):
    # Pattern to match dangerous URL schemes
    dangerous_schemes = r'(?i)(javascript|vbscript|file|about|chrome|ms-\w+):'
    
    # Pattern to match legitimate image data URLs
    image_data_url = r'(?i)data:image/(png|jpg|jpeg|gif|svg\+xml|webp);'
    
    # Check if string contains image data URL
    if re.search(image_data_url, url_string):
        return False
    
    # Check if string contains dangerous schemes
    if re.search(dangerous_schemes, url_string):
        return True
    
    return False
