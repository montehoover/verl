import re
import ast

def evaluate_and_compute(calc_expr: str):
    """
    Evaluates a mathematical expression and returns the result.
    
    Args:
        calc_expr: str, a string containing a mathematical expression to evaluate
        
    Returns:
        The evaluated result of the expression
        
    Raises:
        ValueError: if the expression includes unsupported operators, characters, 
                   unsafe code, or if evaluation fails
    """
    # Remove whitespace
    calc_expr = calc_expr.strip()
    
    # Check for empty expression
    if not calc_expr:
        raise ValueError("Empty expression")
    
    # Validate that the expression only contains allowed characters
    # Allow digits, operators, parentheses, decimal points, and whitespace
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    if not re.match(allowed_pattern, calc_expr):
        raise ValueError("Expression contains unsupported characters")
    
    # Check for potentially unsafe patterns
    if '__' in calc_expr or 'import' in calc_expr or 'eval' in calc_expr or 'exec' in calc_expr:
        raise ValueError("Expression contains unsafe code")
    
    try:
        # Parse the expression into an AST
        node = ast.parse(calc_expr, mode='eval')
        
        # Validate that the AST only contains allowed operations
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                raise ValueError("Function calls are not allowed")
            elif isinstance(n, ast.Name):
                raise ValueError("Variable references are not allowed")
            elif isinstance(n, ast.Attribute):
                raise ValueError("Attribute access is not allowed")
            elif not isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, 
                                   ast.Num, ast.Constant, ast.Add, ast.Sub, 
                                   ast.Mult, ast.Div, ast.Mod, ast.Pow, 
                                   ast.USub, ast.UAdd)):
                raise ValueError(f"Unsupported operation: {type(n).__name__}")
        
        # Evaluate the expression
        result = eval(compile(node, '<string>', 'eval'))
        
        return result
        
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except Exception as e:
        raise ValueError(f"Evaluation failed: {str(e)}")
