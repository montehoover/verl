import ast
import re

def is_safe_expression(expression):
    """
    Check if an expression only contains safe arithmetic operations.
    Returns True if safe, False otherwise.
    """
    # Remove whitespace
    expression = expression.strip()
    
    # Check for empty string
    if not expression:
        return False
    
    # Check for dangerous keywords and constructs
    dangerous_patterns = [
        r'__[a-zA-Z_]+__',  # Dunder methods
        r'\bimport\b',      # Import statements
        r'\beval\b',        # Eval function
        r'\bexec\b',        # Exec function
        r'\bopen\b',        # File operations
        r'\bfile\b',        # File operations
        r'\bos\b',          # OS module
        r'\bsys\b',         # Sys module
        r'\bsubprocess\b',  # Subprocess module
        r'\b(def|class|lambda)\b',  # Function/class definitions
        r'\[.*\]',          # List comprehensions or indexing
        r'\{.*\}',          # Dict/set comprehensions
        r'\.(?![\d])',      # Attribute access (except decimals)
        r'\bfor\b',         # Loops
        r'\bwhile\b',       # Loops
        r'\bif\b',          # Conditionals
        r'\btry\b',         # Exception handling
        r'\bwith\b',        # Context managers
        r';',               # Multiple statements
        r'=',               # Assignment (except ==)
        r'\bprint\b',       # Print function
        r'\binput\b',       # Input function
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, expression):
            # Special case: allow == for comparison
            if pattern == r'=' and not re.search(r'(?<!=)=(?!=)', expression):
                continue
            return False
    
    # Try to parse as AST and check node types
    try:
        tree = ast.parse(expression, mode='eval')
        
        # Define allowed node types
        allowed_nodes = {
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.FloorDiv,
            ast.USub,
            ast.UAdd,
            ast.Constant,  # For Python 3.8+
            ast.Num,       # For older Python versions
            ast.Name,      # Variable names
            ast.Load,      # Loading values
        }
        
        # Walk through all nodes in the AST
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                return False
            
            # Check if Name nodes contain dangerous names
            if isinstance(node, ast.Name):
                if node.id in ['__builtins__', '__import__', 'eval', 'exec', 
                              'compile', 'open', 'input', 'print', 'globals',
                              'locals', 'vars', 'dir', 'getattr', 'setattr',
                              'delattr', 'hasattr', 'callable', 'type']:
                    return False
        
        return True
        
    except (SyntaxError, ValueError):
        return False
