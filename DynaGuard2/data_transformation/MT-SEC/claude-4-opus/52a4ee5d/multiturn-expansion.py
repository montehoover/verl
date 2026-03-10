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

def detect_js_scheme(url_string):
    # Pattern to match harmful schemes (javascript, vbscript, etc.)
    # but exclude data:image URLs
    harmful_pattern = r'(?i)^(?!data:image/)(javascript|vbscript|data):'
    
    return bool(re.search(harmful_pattern, url_string))
