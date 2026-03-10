import re
import ast


def compute_expression(expr: str):
    """
    Evaluates a mathematical expression and returns the result.
    
    Args:
        expr: str, a string containing a mathematical expression to evaluate
        
    Returns:
        The evaluated result of the expression
        
    Raises:
        ValueError: if the expression includes unsupported operators, characters,
                   unsafe code, or if evaluation fails
    """
    # Remove whitespace
    expr = expr.strip()
    
    # Check for empty expression
    if not expr:
        raise ValueError("Empty expression")
    
    # Check for invalid characters - only allow digits, operators, parentheses, dots, and whitespace
    valid_chars_pattern = r'^[0-9+\-*/().\s]+$'
    if not re.match(valid_chars_pattern, expr):
        raise ValueError("Expression contains invalid characters")
    
    # Check for dangerous patterns
    dangerous_patterns = [
        '__', 'import', 'eval', 'exec', 'open', 'file', 'input', 'raw_input'
    ]
    expr_lower = expr.lower()
    
    for pattern in dangerous_patterns:
        if pattern in expr_lower:
            raise ValueError(f"Unsafe pattern '{pattern}' detected")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expr, mode='eval')
        
        # Define allowed node types
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
            ast.USub, ast.UAdd
        )
        
        # Validate all nodes in the AST
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Unsupported operation: {type(node).__name__}")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        
        return result
        
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {str(e)}")
