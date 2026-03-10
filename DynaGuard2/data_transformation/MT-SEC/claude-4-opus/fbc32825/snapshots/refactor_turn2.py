import re

def _extract_text_with_parentheses(text: str) -> tuple[str, str] | None:
    """
    Extract text outside and inside parentheses using regex pattern matching.
    
    Args:
        text: The input string to match against the pattern
        
    Returns:
        tuple[str, str]: A tuple containing (text_outside, text_inside) if pattern matches
        None: If the pattern doesn't match
    """
    pattern = r'^([^(]+)\(([^)]+)\)$'
    match = re.match(pattern, text)
    
    if match:
        return (match.group(1), match.group(2))
    return None


def match_strings(text: str):
    """
    Match a string pattern and capture text outside and inside parentheses.
    
    This function looks for a pattern where text appears before an opening
    parenthesis, followed by text inside parentheses, and ending with a
    closing parenthesis. For example: 'some_text(other_text)'.
    
    Args:
        text: The input string to be matched. Expected format: 'text(content)'
        
    Returns:
        tuple[str, str]: If pattern matches, returns (text_outside, text_inside)
        None: If the pattern doesn't match the expected format
        
    Examples:
        >>> match_strings('some_text(other_text)')
        ('some_text', 'other_text')
        
        >>> match_strings('func(arg1, arg2)')
        ('func', 'arg1, arg2')
        
        >>> match_strings('no_parentheses')
        None
        
        >>> match_strings('empty()')
        None
    """
    return _extract_text_with_parentheses(text)
