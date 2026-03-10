import ast
import re

def parse_user_expression(input_string):
    """
    Parse user input to determine if it's a valid Python expression for basic arithmetic operations.
    
    Args:
        input_string (str): The user's input string
        
    Returns:
        bool: True if the input is a valid and safe arithmetic expression, False otherwise
    """
    # Remove whitespace
    input_string = input_string.strip()
    
    # Check if empty
    if not input_string:
        return False
    
    # Only allow specific characters: digits, operators, parentheses, decimal points, and spaces
    allowed_pattern = r'^[0-9\+\-\*\/\%\(\)\.\s]+$'
    if not re.match(allowed_pattern, input_string):
        return False
    
    # Check for dangerous patterns
    dangerous_patterns = [
        '__', 'import', 'exec', 'eval', 'open', 'file', 'input', 'raw_input',
        'compile', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr',
        'delattr', 'hasattr', 'type', 'help', 'repr', 'print', 'os', 'sys'
    ]
    
    lower_input = input_string.lower()
    for pattern in dangerous_patterns:
        if pattern in lower_input:
            return False
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(input_string, mode='eval')
        
        # Check that the AST only contains allowed node types
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,  # Python 3.8+
            ast.Num,       # Python < 3.8
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.FloorDiv
        )
        
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False
        
        # Try to evaluate to ensure it's valid
        compile(tree, '<string>', 'eval')
        return True
        
    except (SyntaxError, ValueError, TypeError):
        return False
