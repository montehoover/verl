import re
import ast

def validate_math_expression(expression):
    """
    Validates a mathematical expression to ensure it contains only safe mathematical operations.
    
    Args:
        expression (str): The mathematical expression to validate
        
    Returns:
        bool: True if the expression is valid and safe, False otherwise
    """
    if not expression or not isinstance(expression, str):
        return False
    
    # Remove whitespace
    expression = expression.strip()
    
    if not expression:
        return False
    
    # Check for forbidden characters and patterns
    forbidden_patterns = [
        '__',  # Double underscore (dunder methods)
        'import',
        'exec',
        'eval',
        'compile',
        'open',
        'file',
        'input',
        'raw_input',
        'globals',
        'locals',
        'vars',
        'dir',
        'getattr',
        'setattr',
        'delattr',
        'hasattr',
        'callable',
        'classmethod',
        'staticmethod',
        'property',
        'super',
        'type',
        'isinstance',
        'issubclass',
        'lambda',
        'def',
        'class',
        'for',
        'while',
        'if',
        'else',
        'elif',
        'try',
        'except',
        'finally',
        'raise',
        'assert',
        'with',
        'as',
        'yield',
        'from',
        'return',
        'break',
        'continue',
        'pass',
        'del',
        'is',
        'in',
        'not',
        'and',
        'or',
        ';',  # Statement separator
        '\\',  # Escape character
        '`',  # Backtick
        '$',  # Dollar sign
        '{',  # Curly braces
        '}',
        '[',  # Square brackets
        ']',
    ]
    
    expression_lower = expression.lower()
    for pattern in forbidden_patterns:
        if pattern in expression_lower:
            return False
    
    # Allow only specific characters: digits, operators, parentheses, decimal points, and whitespace
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
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
    
    # Try to parse as an AST to ensure it's a valid expression
    try:
        tree = ast.parse(expression, mode='eval')
        
        # Whitelist of allowed AST node types
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Num,  # For Python < 3.8
            ast.Constant,  # For Python >= 3.8
        )
        
        # Check all nodes in the AST
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False
                
        return True
        
    except (SyntaxError, ValueError, TypeError):
        return False
