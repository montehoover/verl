import re
import ast

def evaluate_and_compute(calc_expr: str):
    """
    Evaluates a mathematical expression and returns the result.
    
    Args:
        calc_expr: str - a string containing a mathematical expression to evaluate
        
    Returns:
        The evaluated result of the expression
        
    Raises:
        ValueError: if the expression includes unsupported operators, characters, 
                   unsafe code, or if evaluation fails
    """
    # Remove whitespace for easier validation
    cleaned_expr = calc_expr.strip()
    
    # Check if expression is empty
    if not cleaned_expr:
        raise ValueError("Empty expression provided")
    
    # Define allowed characters and operators
    allowed_chars = r'^[0-9\+\-\*\/\(\)\.\s]+$'
    
    # Validate that only allowed characters are present
    if not re.match(allowed_chars, cleaned_expr):
        raise ValueError("Expression contains unsupported characters or operators")
    
    try:
        # Parse the expression into an AST
        node = ast.parse(cleaned_expr, mode='eval')
        
        # Define allowed node types for safe evaluation
        allowed_nodes = {
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Constant,  # For Python 3.8+
            ast.Num,       # For Python < 3.8 compatibility
        }
        
        # Validate that all nodes in the AST are allowed
        for node_type in ast.walk(node):
            if type(node_type) not in allowed_nodes:
                # Special handling for Name nodes (variables) - not allowed
                if isinstance(node_type, ast.Name):
                    raise ValueError("Variables are not allowed in expressions")
                # Special handling for Call nodes (function calls) - not allowed
                elif isinstance(node_type, ast.Call):
                    raise ValueError("Function calls are not allowed in expressions")
                # Any other node type is also not allowed
                elif type(node_type) not in allowed_nodes:
                    raise ValueError(f"Unsupported operation: {type(node_type).__name__}")
        
        # Compile and evaluate the expression
        compiled = compile(node, '<string>', 'eval')
        result = eval(compiled)
        
        return result
        
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except ZeroDivisionError:
        raise ValueError("Division by zero error")
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {str(e)}")
