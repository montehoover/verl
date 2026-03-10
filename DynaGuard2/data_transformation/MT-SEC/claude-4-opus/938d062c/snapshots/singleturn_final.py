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
        raise ValueError("Empty expression")
    
    # Check for invalid characters using regex
    # Allow digits, operators (+, -, *, /, **, //, %), parentheses, dots (for decimals), and whitespace
    if not re.match(r'^[0-9+\-*/%(). ]+$', math_expr):
        raise ValueError("Expression contains invalid characters")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(math_expr, mode='eval')
        
        # Check that the AST only contains allowed node types
        for node in ast.walk(tree):
            # Allow only safe node types for mathematical expressions
            allowed_types = (
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
                ast.FloorDiv, ast.USub, ast.UAdd
            )
            
            # For Python < 3.8 compatibility, also allow Num node type
            if hasattr(ast, 'Num'):
                allowed_types = allowed_types + (ast.Num,)
            
            if not isinstance(node, allowed_types):
                raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        
        # Check for division by zero (results in inf or -inf)
        if result == float('inf') or result == float('-inf'):
            raise ValueError("Division by zero")
        
        # Return integer if result is a whole number, otherwise float
        if isinstance(result, float) and result.is_integer():
            return int(result)
        
        return result
        
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {str(e)}")
