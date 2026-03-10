import re

def analyze_scheme(url: str) -> bool:
    """
    Analyzes a given string to determine if it contains a potentially harmful URL scheme.
    
    Args:
        url: str, the string to examine for possible dangerous URL schemes.
        
    Returns:
        bool, returns True if the string involves a potentially harmful URL scheme,
        aside from valid image data URLs; returns False otherwise.
    """
    # Pattern for dangerous URL schemes
    dangerous_schemes = r'^(javascript|vbscript|data:(?!image/)[^;]+|about|chrome|chrome-extension|ms-help|hcp|mk|mhtml|help|disk|afp|x-man-page|x-help|ms-its|ms-itss|ldap|mailto|news|nntp|telnet|gopher|wais|file):'
    
    # Pattern for valid image data URLs
    image_data_pattern = r'^data:image/(gif|jpeg|jpg|png|svg\+xml|webp);'
    
    # Check if it's a valid image data URL first
    if re.match(image_data_pattern, url, re.IGNORECASE):
        return False
    
    # Check for dangerous schemes
    if re.match(dangerous_schemes, url, re.IGNORECASE):
        return True
    
    return False
