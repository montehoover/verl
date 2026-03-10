import re

def is_javascript_scheme(s: str) -> bool:
    """
    Checks if a given string contains a possibly malicious URL scheme
    (e.g., JavaScript, jscript, vbscript, data: (non-image)).
    Excludes image data URLs from these checks.

    Args:
        s: str, the string to be checked for possibly malicious URL schemes.

    Returns:
        bool: Returns True if the string contains a possibly malicious URL scheme
              (excluding image data URLs); otherwise, False.
    
    Raises:
        It does not explicitly raise exceptions but improper inputs 
        (e.g., non-string type for s, though type-hinted as str) 
        or faulty regex patterns could potentially raise exceptions 
        like 're.error' or AttributeError if s.strip() is called on non-str.
    """
    # Assuming s is a string as per type hint.
    # If s could be None or other types, additional checks like:
    # if not isinstance(s, str): return False # or raise TypeError
    # would be necessary before calling .strip(). For this implementation,
    # we rely on the caller to provide a string.

    normalized_s = s.strip().lower()

    # 1. Exclude image data URLs (these are considered safe in this context).
    #    Pattern: starts with "data:image/"
    #    Example: "data:image/png;base64,..." -> False
    if re.match(r'^data:image\/', normalized_s):
        return False

    # 2. Check for explicit script-like schemes (javascript, jscript, vbscript).
    #    Pattern: starts with "javascript:", "jscript:", or "vbscript:"
    #    Example: "javascript:alert('XSS')" -> True
    if re.match(r'^(javascript|jscript|vbscript):', normalized_s):
        return True

    # 3. Check for other "data:" URLs (non-image).
    #    The example "data:;base64,..." -> True implies these are treated as risky.
    #    Pattern: starts with "data:" (and not caught by rule 1 as an image data URL)
    #    Example: "data:text/html,<script>alert(0)</script>" -> True
    #    Example from prompt: "data:;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA" -> True
    if re.match(r'^data:', normalized_s):
        return True
        
    # If none of the above conditions are met, the scheme is not considered malicious.
    return False
