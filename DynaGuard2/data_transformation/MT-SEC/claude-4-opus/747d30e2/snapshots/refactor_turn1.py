import re

def is_javascript_scheme(s):
    # Pattern to match potentially malicious URL schemes
    malicious_schemes_pattern = r'(?:javascript|jscript|vbscript|mocha|livescript|ecmascript|about)\s*:'
    
    # Pattern to match image data URLs (these are safe)
    image_data_pattern = r'data:image/(?:png|jpg|jpeg|gif|bmp|webp|svg\+xml|x-icon|vnd\.microsoft\.icon);'
    
    # First check if it's an image data URL (which is safe)
    if re.search(image_data_pattern, s, re.IGNORECASE):
        return False
    
    # Then check for malicious schemes
    if re.search(malicious_schemes_pattern, s, re.IGNORECASE):
        return True
    
    return False
