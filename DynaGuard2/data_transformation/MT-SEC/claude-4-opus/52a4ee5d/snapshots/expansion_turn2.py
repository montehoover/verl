import re

def extract_url_candidates(text):
    # Regex pattern to match URL-like patterns
    url_pattern = r'https?://[^\s<>"{}|\\^\[\]`]+|www\.[^\s<>"{}|\\^\[\]`]+\.[^\s<>"{}|\\^\[\]`]+'
    
    # Find all matches
    matches = re.findall(url_pattern, text)
    
    return matches

def identify_url_scheme(url_candidate):
    # Regex pattern to match URL scheme
    scheme_pattern = r'^([a-zA-Z][a-zA-Z0-9+.-]*):\/\/'
    
    match = re.match(scheme_pattern, url_candidate)
    
    if match:
        return match.group(1).lower()
    else:
        return 'unknown'
