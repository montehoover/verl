import re
import ast

def parse_and_calculate(expression: str):
    """
    Processes a mathematical expression provided by a user, parsing the expression 
    and performing the calculation based on the input.
    
    Args:
        expression: str, the arithmetic expression provided by the user
        
    Returns:
        The result of the calculated expression
        
    Raises:
        ValueError: if the input contains unsupported characters, malicious commands, 
                   or cannot compute
    """
    # Remove whitespace for easier validation
    cleaned_expr = expression.strip()
    
    # Check if expression is empty
    if not cleaned_expr:
        raise ValueError("Empty expression provided")
    
    # Validate that the expression only contains allowed characters
    # Allowed: digits, operators (+, -, *, /, **, //, %), parentheses, decimal points, and whitespace
    allowed_pattern = r'^[\d\s\+\-\*\/\%\(\)\.]+$'
    if not re.match(allowed_pattern, cleaned_expr):
        raise ValueError("Expression contains unsupported characters")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(cleaned_expr, mode='eval')
        
        # Validate that the AST only contains allowed node types
        for node in ast.walk(tree):
            # Allow only safe node types for arithmetic expressions
            allowed_types = (
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
                ast.Num,  # For Python < 3.8 compatibility
                ast.Constant,  # For Python >= 3.8
            )
            if not isinstance(node, allowed_types):
                raise ValueError(f"Unsupported operation in expression")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        
        # Check for division by zero or other arithmetic errors that result in inf/nan
        if isinstance(result, float):
            if result == float('inf') or result == float('-inf'):
                raise ValueError("Division by zero or arithmetic overflow")
            if result != result:  # NaN check
                raise ValueError("Invalid arithmetic operation")
        
        return result
        
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Cannot compute expression: {str(e)}")
