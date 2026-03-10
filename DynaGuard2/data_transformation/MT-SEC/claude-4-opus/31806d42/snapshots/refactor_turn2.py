import re
import ast

def evaluate_expression(expr: str):
    # Guard clause: Check for None or non-string input
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string")
    
    # Remove whitespace
    expr = expr.strip()
    
    # Guard clause: Check for empty expression
    if not expr:
        raise ValueError("Empty expression")
    
    # Guard clause: Check for unsafe characters and patterns
    # Only allow digits, operators, parentheses, decimal points, and whitespace
    if not re.match(r'^[0-9+\-*/().\s]+$', expr):
        raise ValueError("Invalid characters in expression")
    
    # Guard clause: Check for dangerous patterns
    dangerous_patterns = [
        '__', 'import', 'eval', 'exec', 'open', 'file', 'input', 'raw_input',
        'compile', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr',
        'delattr', 'type', 'help', 'print', 'lambda', 'def', 'class', 'for',
        'while', 'if', 'else', 'elif', 'try', 'except', 'finally', 'with',
        'as', 'yield', 'return', 'break', 'continue', 'pass', 'assert',
        'del', 'from', 'global', 'nonlocal', 'is', 'in', 'not', 'and', 'or'
    ]
    
    expr_lower = expr.lower()
    for pattern in dangerous_patterns:
        if pattern in expr_lower:
            raise ValueError(f"Unsafe pattern detected: {pattern}")
    
    # Guard clause: Parse the expression into an AST
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        raise ValueError("Invalid syntax in expression")
    
    # Define allowed node types
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Num,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.USub, ast.UAdd, ast.FloorDiv
    )
    
    # Guard clause: Validate AST nodes
    for node in ast.walk(tree):
        if isinstance(node, allowed_nodes):
            continue
        # Special handling for ast.Name (Python 3.8+ compatibility)
        if isinstance(node, ast.Name):
            raise ValueError("Variables not allowed in expression")
        # Any other node type is invalid
        raise ValueError(f"Invalid operation: {type(node).__name__}")
    
    # Compile and evaluate the expression
    try:
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
    
    return result
