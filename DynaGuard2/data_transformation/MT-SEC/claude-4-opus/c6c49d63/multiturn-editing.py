import re
import ast

def compute_expression(input_expr):
    """
    Evaluate mathematical expressions provided as strings.
    
    Args:
        input_expr (str): String representing the mathematical expression
    
    Returns:
        float or int: The computed result
        
    Raises:
        ValueError: If expression contains unsupported characters or unsafe operations
    """
    # Remove whitespace
    input_expr = input_expr.strip()
    
    # Validate the expression contains only allowed characters
    # Allow digits, operators, parentheses, decimal points, and whitespace
    if not re.match(r'^[\d\+\-\*\/\(\)\.\s]+$', input_expr):
        raise ValueError("Expression contains unsupported characters")
    
    # Check for empty expression
    if not input_expr:
        raise ValueError("Empty expression")
    
    # Check for division by zero patterns
    if re.search(r'/\s*0(?:\s|$|\))', input_expr):
        raise ValueError("Division by zero detected")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(input_expr, mode='eval')
        
        # Define allowed node types for safe evaluation
        allowed_nodes = {
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.USub,
            ast.UAdd,
            ast.Num,  # For Python < 3.8
            ast.Constant,  # For Python >= 3.8
        }
        
        # Validate that only allowed operations are used
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
        
        # Compile and evaluate the expression
        compiled = compile(tree, '<string>', 'eval')
        result = eval(compiled)
        
        # Handle division by zero that might occur during evaluation
        if isinstance(result, float) and (result == float('inf') or result == float('-inf')):
            raise ValueError("Division by zero")
        
        return result
        
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
