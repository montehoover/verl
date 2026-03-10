import re
import ast

def evaluate_expression(expr: str):
    # Remove whitespace
    expr = expr.strip()
    
    # Check for empty expression
    if not expr:
        raise ValueError("Empty expression")
    
    # Check for unsafe characters and patterns
    # Only allow digits, operators, parentheses, decimal points, and whitespace
    if not re.match(r'^[0-9+\-*/().\s]+$', expr):
        raise ValueError("Invalid characters in expression")
    
    # Check for dangerous patterns
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
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expr, mode='eval')
        
        # Check that the AST only contains allowed node types
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Num,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
            ast.USub, ast.UAdd, ast.FloorDiv
        )
        
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                # Handle ast.Name for Python 3.8+ compatibility with constants
                if isinstance(node, ast.Name):
                    raise ValueError(f"Variables not allowed in expression")
                else:
                    raise ValueError(f"Invalid operation: {type(node).__name__}")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        
        return result
        
    except SyntaxError:
        raise ValueError("Invalid syntax in expression")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
