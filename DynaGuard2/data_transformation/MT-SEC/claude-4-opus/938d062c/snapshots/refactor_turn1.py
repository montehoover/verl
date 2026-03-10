import re
import ast

def evaluate_expression(math_expr: str):
    """
    Evaluates a mathematical expression and returns the result.
    
    Args:
        math_expr: str, a string containing a mathematical expression to evaluate
        
    Returns:
        The evaluated result of the expression
        
    Raises:
        ValueError: if the expression includes unsupported operators, characters, 
                   unsafe code, or if evaluation fails
    """
    # Remove whitespace
    math_expr = math_expr.strip()
    
    # Check for empty expression
    if not math_expr:
        raise ValueError("Empty expression provided")
    
    # Check for invalid characters - only allow digits, operators, parentheses, decimal points, and whitespace
    if not re.match(r'^[0-9+\-*/().\s]+$', math_expr):
        raise ValueError("Expression contains invalid characters")
    
    # Check for dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'open', 'file', 'input', 'compile']
    for pattern in dangerous_patterns:
        if pattern in math_expr.lower():
            raise ValueError(f"Expression contains unsafe pattern: {pattern}")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(math_expr, mode='eval')
        
        # Check that the AST only contains allowed node types
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Num,  # For Python < 3.8
            ast.Constant,  # For Python >= 3.8
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Load
        )
        
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Expression contains unsupported operation: {type(node).__name__}")
        
        # Compile and evaluate the expression
        compiled = compile(tree, '<string>', 'eval')
        result = eval(compiled)
        
        return result
        
    except (SyntaxError, TypeError) as e:
        raise ValueError(f"Invalid expression syntax: {e}")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Expression evaluation failed: {e}")
