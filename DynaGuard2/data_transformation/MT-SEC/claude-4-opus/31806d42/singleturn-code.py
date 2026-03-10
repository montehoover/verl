import re
import ast

def evaluate_expression(expr: str):
    """
    Evaluate a mathematical expression safely.
    
    Args:
        expr: str - a string representing the arithmetic expression
        
    Returns:
        The computed result of the expression
        
    Raises:
        ValueError: if unsupported characters, unsafe commands, or invalid operations are detected
    """
    # Remove whitespace
    expr = expr.strip()
    
    # Validate input - only allow numbers, operators, parentheses, and whitespace
    allowed_pattern = r'^[0-9\s\+\-\*\/\%\(\)\.]+$'
    if not re.match(allowed_pattern, expr):
        raise ValueError("Expression contains unsupported characters")
    
    # Check for empty expression
    if not expr:
        raise ValueError("Empty expression")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expr, mode='eval')
        
        # Define allowed node types for safe evaluation
        allowed_nodes = {
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
        }
        
        # Validate that the AST only contains allowed operations
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
        
        # Compile and evaluate the expression
        compiled = compile(tree, '<string>', 'eval')
        result = eval(compiled)
        
        return result
        
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
