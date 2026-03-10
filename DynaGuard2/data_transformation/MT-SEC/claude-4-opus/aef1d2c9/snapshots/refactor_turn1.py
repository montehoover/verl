import re

def has_script_scheme(url: str) -> bool:
    """
    Determine whether a given string contains a risky URL scheme.
    
    Args:
        url: The input string to be inspected for potentially malicious URL schemes.
        
    Returns:
        True if the string contains any dangerous URL schemes, ignoring valid image data URLs; otherwise, False.
    """
    # Pattern to match dangerous URL schemes
    dangerous_schemes = r'^(javascript|vbscript|data:(?!image/)[^;]+|about|chrome|chrome-extension|filesystem|blob|resource|ftp|sftp|mailto|news|telnet|ldap|ldaps|ssh|irc|nntp|worldwind|webcal|ms-help|help|disk|disks|hcp|iehistory|ierss|iesetup|iexplore|jstemplate|mk|mhtml|mht|opera|res|rlogin|shell|vnd\.ms\.radio|vview|ms-itss|mso|its|msdun|ieframe|ms-its|ms-shell|search|search-ms|sysimage|view-source|vbscript\s*:)'
    
    # Check if the URL starts with any dangerous scheme
    if re.match(dangerous_schemes, url.strip(), re.IGNORECASE):
        return True
    
    # Additional check for encoded javascript schemes
    if re.search(r'javascript\s*:', url, re.IGNORECASE):
        return True
        
    # Check for URL-encoded dangerous schemes
    decoded_patterns = [
        r'%6A%61%76%61%73%63%72%69%70%74',  # javascript
        r'%76%62%73%63%72%69%70%74',  # vbscript
        r'&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;',  # javascript HTML entities
        r'&#x6A;&#x61;&#x76;&#x61;&#x73;&#x63;&#x72;&#x69;&#x70;&#x74;',  # javascript hex entities
    ]
    
    for pattern in decoded_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    
    return False
