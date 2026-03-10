import re

def find_urls(text):
    # Pattern to match URL-like strings with capturing groups for scheme
    # Captures the scheme (http://, https://, ftp://) or www. as separate groups
    url_pattern = r'((?:https?|ftp)://)([^\s]+)|(www\.)([^\s]+)'
    
    # Find all matches in the text
    matches = re.findall(url_pattern, text)
    
    # Process matches to extract URLs and their schemes
    results = []
    for match in matches:
        if match[0]:  # Has explicit scheme (http://, https://, ftp://)
            scheme = match[0].rstrip('://')
            url = match[0] + match[1]
        else:  # www. pattern (no explicit scheme)
            scheme = 'www'
            url = match[2] + match[3]
        results.append((url, scheme))
    
    return results

def contains_dangerous_scheme(user_input):
    # Pattern to match dangerous URL schemes like javascript:, vbscript:, etc.
    # Case-insensitive matching for various dangerous schemes
    dangerous_schemes_pattern = r'(?i)(javascript|vbscript|file|about|chrome|res|jar|ms-its|mhtml|data):'
    
    # Pattern to match legitimate image data URLs
    image_data_url_pattern = r'(?i)data:image/(png|jpg|jpeg|gif|svg\+xml|webp);base64,'
    
    try:
        # First check if there are any image data URLs and remove them from consideration
        cleaned_input = re.sub(image_data_url_pattern, '', user_input)
        
        # Now check for dangerous schemes in the cleaned input
        if re.search(dangerous_schemes_pattern, cleaned_input):
            # Additional check to ensure 'data:' URLs that remain are dangerous
            if 'data:' in cleaned_input.lower():
                # Check if remaining data: URLs are non-image types
                remaining_data_matches = re.findall(r'(?i)data:[^;]+', cleaned_input)
                for match in remaining_data_matches:
                    if not re.match(r'(?i)data:image/', match):
                        return True
            else:
                return True
        
        return False
    except:
        # If regex fails, default to safe behavior
        return False
