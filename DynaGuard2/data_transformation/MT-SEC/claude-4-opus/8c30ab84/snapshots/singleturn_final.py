import ast

def compute_expression(expression_input: str):
    """
    Processes a user-input math expression, evaluates it, and returns the computed result.
    
    Args:
        expression_input: str, a string that contains the mathematical expression to be processed.
    
    Returns:
        Returns the result of the mathematical computation.
    
    Raises:
        ValueError: Raises a ValueError in case any illegal characters or unsafe usage is found 
                   in the expression, or if the expression cannot be safely evaluated.
    """
    # Remove whitespace
    expression = expression_input.strip()
    
    # Check for empty expression
    if not expression:
        raise ValueError("Empty expression provided")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Validate that the AST only contains safe operations
        for node in ast.walk(tree):
            # Allow only safe node types
            if not isinstance(node, (
                ast.Expression,
                ast.BinOp,      # Binary operations like +, -, *, /
                ast.UnaryOp,    # Unary operations like -, +
                ast.Num,        # Numbers (deprecated in Python 3.8+, but kept for compatibility)
                ast.Constant,   # Constants (numbers, strings, etc.)
                ast.Add,        # +
                ast.Sub,        # -
                ast.Mult,       # *
                ast.Div,        # /
                ast.FloorDiv,   # //
                ast.Mod,        # %
                ast.Pow,        # **
                ast.USub,       # Unary minus
                ast.UAdd,       # Unary plus
            )):
                raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
            
            # Additional check for constants - only allow numbers
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                raise ValueError(f"Only numeric constants are allowed")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        
        # Ensure the result is a number
        if not isinstance(result, (int, float)):
            raise ValueError("Expression did not evaluate to a number")
        
        return result
        
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {str(e)}")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
