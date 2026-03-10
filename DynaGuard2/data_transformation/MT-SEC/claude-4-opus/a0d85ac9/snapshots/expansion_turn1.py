import re
import ast

def parse_expression(expression):
    """
    Parse and validate a user-supplied expression to ensure it contains only
    valid arithmetic or logical operations.
    
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
    
    # Define allowed characters and patterns
    allowed_pattern = re.compile(r'^[0-9\s\+\-\*\/\%\(\)\.\,\&\|\^\~\<\>\=\!]+$')
    
    # Check if expression contains only allowed characters
    if not allowed_pattern.match(expression):
        return False
    
    # Blacklist of dangerous keywords/functions
    dangerous_keywords = [
        '__import__', 'eval', 'exec', 'compile', 'open', 'file', 'input',
        'raw_input', 'execfile', 'getattr', 'setattr', 'delattr', 'vars',
        'globals', 'locals', 'reload', 'import', 'from', 'as', 'lambda',
        'def', 'class', 'with', 'yield', 'raise', 'try', 'except', 'finally',
        'assert', 'del', 'pass', 'break', 'continue', 'return', 'global',
        'nonlocal', 'async', 'await', 'print', 'help', 'dir', 'type',
        'isinstance', 'issubclass', 'super', 'property', 'staticmethod',
        'classmethod', 'callable', 'format', 'repr', 'ascii', 'ord', 'chr',
        'bin', 'hex', 'oct', 'abs', 'round', 'divmod', 'pow', 'sorted',
        'reversed', 'enumerate', 'filter', 'map', 'zip', 'all', 'any',
        'sum', 'min', 'max', 'len', 'range', 'slice', 'iter', 'next',
        'id', 'hash', 'bytes', 'bytearray', 'memoryview', 'complex',
        'bool', 'list', 'tuple', 'set', 'frozenset', 'dict', 'str',
        'int', 'float', 'object', 'subprocess', 'os', 'sys', 'shutil'
    ]
    
    # Check for dangerous keywords
    expression_lower = expression.lower()
    for keyword in dangerous_keywords:
        if keyword in expression_lower:
            return False
    
    # Check for double underscores (often used in Python internals)
    if '__' in expression:
        return False
    
    # Check for attribute access or method calls
    if '.' in expression:
        return False
    
    # Check for string literals (could be used for code injection)
    if '"' in expression or "'" in expression or '`' in expression:
        return False
    
    # Try to parse as AST to validate structure
    try:
        tree = ast.parse(expression, mode='eval')
        
        # Define allowed node types for arithmetic and logical operations
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
            ast.Constant, ast.Num, ast.Name, ast.Load, ast.Add, ast.Sub,
            ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.LShift,
            ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.And, ast.Or,
            ast.Not, ast.Invert, ast.UAdd, ast.USub, ast.Eq, ast.NotEq,
            ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In,
            ast.NotIn
        )
        
        # Check all nodes in the AST
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False
            
            # Additional check for Name nodes - only allow specific variable names
            if isinstance(node, ast.Name):
                allowed_names = ['True', 'False', 'None']
                if node.id not in allowed_names:
                    # Allow single letter variables or numeric-like names
                    if not (len(node.id) == 1 and node.id.isalpha()) and not node.id.replace('_', '').isalnum():
                        return False
        
        return True
        
    except (SyntaxError, ValueError, TypeError):
        return False
