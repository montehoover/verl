import re

def find_urls(text):
    # Pattern to match URL-like strings with capturing group for scheme
    url_pattern = r'(https?)://[^\s<>"{}|\\^\[\]`]+'
    
    # Find all matches in the text
    matches = re.findall(url_pattern, text)
    
    # Create list of tuples with full URL and scheme
    urls = []
    for match in re.finditer(url_pattern, text):
        full_url = match.group(0)
        scheme = match.group(1)
        urls.append((full_url, scheme))
    
    return urls

def check_javascript_scheme(url_str):
    # Pattern to match dangerous URL schemes
    dangerous_pattern = r'^(javascript|vbscript):'
    
    # Pattern to match safe image data URLs
    safe_image_pattern = r'^data:image/(png|jpg|jpeg|gif|svg\+xml|webp);'
    
    # Check if it's a safe image data URL first
    if re.match(safe_image_pattern, url_str, re.IGNORECASE):
        return False
    
    # Check for dangerous schemes
    if re.match(dangerous_pattern, url_str, re.IGNORECASE):
        return True
    
    return False
