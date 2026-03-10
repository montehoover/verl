import re
import ast

def parse_quiz_expression(expression):
    """
    Parse a user-submitted quiz expression and determine if it's valid for evaluation.
    
    Args:
        expression (str): The expression to validate
        
    Returns:
        bool: True if the expression is valid and safe, False otherwise
    """
    if not expression or not isinstance(expression, str):
        return False
    
    # Remove whitespace
    expression = expression.strip()
    
    # Check for empty expression
    if not expression:
        return False
    
    # Define allowed characters and patterns for quiz expressions
    # Allow numbers, basic operators, parentheses, and spaces
    allowed_pattern = r'^[0-9\+\-\*\/\(\)\.\s]+$'
    
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
    
    try:
        # Use ast.parse to check if it's a valid expression
        tree = ast.parse(expression, mode='eval')
        
        # Walk the AST to ensure only safe operations
        for node in ast.walk(tree):
            # Allow only safe node types
            allowed_types = (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Constant,  # Python 3.8+
                ast.Num,       # For older Python versions
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.USub,
                ast.UAdd,
                ast.Load
            )
            
            if not isinstance(node, allowed_types):
                return False
        
        # Try to compile the expression to catch any syntax errors
        compile(tree, '<string>', 'eval')
        return True
        
    except (SyntaxError, ValueError, TypeError):
        return False
