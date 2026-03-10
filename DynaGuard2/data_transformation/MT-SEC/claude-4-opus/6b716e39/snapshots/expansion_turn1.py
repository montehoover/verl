import re

def validate_expression(expression):
    """
    Validates if a string contains only valid arithmetic or logical operations.
    Returns True if valid, False otherwise.
    """
    # Check for empty or None input
    if not expression or not isinstance(expression, str):
        return False
    
    # Remove whitespace for analysis
    expression = expression.strip()
    
    # Empty expression after stripping
    if not expression:
        return False
    
    # Define allowed characters and patterns
    # Numbers, operators, parentheses, and common math functions
    allowed_pattern = r'^[0-9\+\-\*/\(\)\s\.\,\^\%\&\|\!\=\<\>]+$'
    
    # Check for dangerous keywords that could indicate code injection
    dangerous_keywords = [
        'import', 'exec', 'eval', '__', 'open', 'file', 'input',
        'compile', 'globals', 'locals', 'vars', 'dir', 'help',
        'type', 'isinstance', 'getattr', 'setattr', 'delattr',
        'hasattr', 'callable', 'classmethod', 'staticmethod',
        'property', 'super', 'object', 'print', 'exit', 'quit',
        'os', 'sys', 'subprocess', 'pickle', 'shelve', 'marshal',
        'importlib', 'runpy', 'execfile', 'raw_input', 'reload',
        'del', 'lambda', 'def', 'class', 'with', 'as', 'yield',
        'return', 'break', 'continue', 'pass', 'raise', 'try',
        'except', 'finally', 'assert', 'global', 'nonlocal',
        'from', 'import', 'for', 'while', 'if', 'elif', 'else'
    ]
    
    # Convert to lowercase for case-insensitive checking
    expression_lower = expression.lower()
    
    # Check for dangerous keywords
    for keyword in dangerous_keywords:
        if keyword in expression_lower:
            return False
    
    # Check if expression matches allowed pattern
    if not re.match(allowed_pattern, expression):
        return False
    
    # Check for balanced parentheses
    paren_count = 0
    for char in expression:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        if paren_count < 0:
            return False
    
    if paren_count != 0:
        return False
    
    # Additional check for consecutive operators (except for negative numbers)
    operator_pattern = r'[\+\-\*/\^%]{2,}'
    if re.search(operator_pattern, expression):
        # Allow -- for double negative, but not other combinations
        cleaned = re.sub(r'--', '', expression)
        if re.search(operator_pattern, cleaned):
            return False
    
    return True
