import ast
import re

def parse_user_query(query):
    """
    Parse user query to determine if it's a valid arithmetic expression.
    Returns True if valid, False otherwise.
    """
    # Remove whitespace
    query = query.strip()
    
    # Check if empty
    if not query:
        return False
    
    # Check for potentially harmful patterns
    dangerous_patterns = [
        '__', 'import', 'exec', 'eval', 'open', 'file', 'input',
        'compile', 'globals', 'locals', 'vars', 'dir', 'help',
        'type', 'isinstance', 'getattr', 'setattr', 'delattr',
        'hasattr', 'callable', 'classmethod', 'staticmethod',
        'property', 'super', 'object', 'lambda', 'def', 'class',
        'for', 'while', 'if', 'else', 'elif', 'try', 'except',
        'finally', 'raise', 'assert', 'with', 'as', 'yield',
        'from', 'return', 'break', 'continue', 'pass', 'del',
        'is', 'in', 'not', 'and', 'or', 'True', 'False', 'None'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(r'\b' + pattern + r'\b', query, re.IGNORECASE):
            return False
    
    # Check for function calls (anything followed by parentheses)
    if re.search(r'[a-zA-Z_]\w*\s*\(', query):
        return False
    
    # Check for attribute access
    if '.' in query:
        return False
    
    # Check for list/dict comprehensions or indexing
    if '[' in query or ']' in query or '{' in query or '}' in query:
        return False
    
    try:
        # Parse the query as an AST
        tree = ast.parse(query, mode='eval')
        
        # Define allowed node types for arithmetic expressions
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Constant,  # For Python 3.8+
            ast.Num,       # For older Python versions
            ast.Name,
        )
        
        # Check if all nodes in the AST are allowed
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False
            
            # If it's a Name node, only allow specific variable names
            if isinstance(node, ast.Name):
                # You can customize allowed variable names here
                allowed_vars = ['x', 'y', 'z', 'a', 'b', 'c']
                if node.id not in allowed_vars:
                    return False
        
        return True
        
    except (SyntaxError, ValueError, TypeError):
        return False
