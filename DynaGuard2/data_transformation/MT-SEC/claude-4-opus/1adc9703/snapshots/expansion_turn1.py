import re
import ast

def parse_expression(expression):
    """
    Parse a mathematical expression and determine if it's valid and safe.
    
    Args:
        expression (str): The mathematical expression to parse
        
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
    # Allow only numbers, operators, parentheses, and decimal points
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    if not re.match(allowed_pattern, expression):
        return False
    
    # Check for dangerous patterns
    dangerous_patterns = [
        r'__',  # Double underscore (could be trying to access special attributes)
        r'import',  # Import statements
        r'exec',  # Exec function
        r'eval',  # Eval function
        r'compile',  # Compile function
        r'open',  # File operations
        r'file',  # File operations
        r'input',  # Input function
        r'raw_input',  # Raw input function
        r'\[',  # List comprehensions
        r'\]',  # List comprehensions
        r'{',  # Dictionary/set comprehensions
        r'}',  # Dictionary/set comprehensions
        r'lambda',  # Lambda functions
        r'def',  # Function definitions
        r'class',  # Class definitions
        r'for',  # Loops
        r'while',  # Loops
        r'if',  # Conditionals
        r'else',  # Conditionals
        r'elif',  # Conditionals
        r'try',  # Exception handling
        r'except',  # Exception handling
        r'raise',  # Exception handling
        r'assert',  # Assertions
        r'global',  # Global declarations
        r'nonlocal',  # Nonlocal declarations
        r'del',  # Delete statements
        r'with',  # Context managers
        r'yield',  # Generators
        r'return',  # Return statements
        r'pass',  # Pass statements
        r'break',  # Break statements
        r'continue',  # Continue statements
    ]
    
    expression_lower = expression.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, expression_lower):
            return False
    
    # Check for valid parentheses matching
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
    
    # Check for empty parentheses
    if '()' in expression:
        return False
    
    # Check for multiple consecutive operators
    if re.search(r'[+\-*/]{2,}', expression):
        # Allow -- and ++ as they could be double negation or positive
        if not re.match(r'^[+\-]+$', re.search(r'[+\-*/]{2,}', expression).group()):
            return False
    
    # Try to parse as AST to check structure
    try:
        tree = ast.parse(expression, mode='eval')
        
        # Check that the AST only contains allowed node types
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Num,  # For Python < 3.8
            ast.Constant,  # For Python >= 3.8
        )
        
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False
    except:
        return False
    
    return True
