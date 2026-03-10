import re
import ast

def parse_quiz_expression(expression):
    """
    Parse a user-submitted quiz expression and determine if it's valid for evaluation.
    
    Args:
        expression (str): The expression to validate
        
    Returns:
        bool: True if the expression is valid and safe, False otherwise
    """
    if not expression or not isinstance(expression, str):
        return False
    
    # Remove whitespace
    expression = expression.strip()
    
    # Check for empty expression
    if not expression:
        return False
    
    # Define allowed characters and patterns for quiz expressions
    # Allow numbers, basic operators, parentheses, and spaces
    allowed_pattern = r'^[0-9\+\-\*\/\(\)\.\s]+$'
    
    if not re.match(allowed_pattern, expression):
        return False
    
    # Check for balanced parentheses
    paren_count = 0
    for char in expression:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        if paren_count < 0:
            return False
    
    if paren_count != 0:
        return False
    
    try:
        # Use ast.parse to check if it's a valid expression
        tree = ast.parse(expression, mode='eval')
        
        # Walk the AST to ensure only safe operations
        for node in ast.walk(tree):
            # Allow only safe node types
            allowed_types = (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Constant,  # Python 3.8+
                ast.Num,       # For older Python versions
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.USub,
                ast.UAdd,
                ast.Load
            )
            
            if not isinstance(node, allowed_types):
                return False
        
        # Try to compile the expression to catch any syntax errors
        compile(tree, '<string>', 'eval')
        return True
        
    except (SyntaxError, ValueError, TypeError):
        return False


def safe_evaluate_expression(expression):
    """
    Safely evaluate a validated expression and return the result as a string.
    
    Args:
        expression (str): The expression to evaluate
        
    Returns:
        str: The result of the evaluation or an error message
    """
    if not isinstance(expression, str):
        return "Error: Expression must be a string"
    
    # First validate the expression
    if not parse_quiz_expression(expression):
        return "Error: Invalid or unsafe expression"
    
    try:
        # Parse the expression
        tree = ast.parse(expression, mode='eval')
        
        # Double-check that only safe operations are present
        for node in ast.walk(tree):
            allowed_types = (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Constant,
                ast.Num,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.USub,
                ast.UAdd,
                ast.Load
            )
            
            if not isinstance(node, allowed_types):
                return "Error: Unsafe operation detected"
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        
        # Create a restricted environment for evaluation
        safe_dict = {
            '__builtins__': {},
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
        }
        
        result = eval(code, safe_dict)
        
        # Handle division by zero
        if isinstance(result, float) and (result == float('inf') or result == float('-inf')):
            return "Error: Division by zero"
        
        # Handle NaN
        if isinstance(result, float) and result != result:  # NaN check
            return "Error: Invalid mathematical operation"
        
        # Format the result
        if isinstance(result, float):
            # Check if it's a whole number
            if result.is_integer():
                return str(int(result))
            else:
                # Format to reasonable precision
                return f"{result:.10g}"
        else:
            return str(result)
            
    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: Invalid value - {str(e)}"
    except SyntaxError:
        return "Error: Invalid expression syntax"
    except Exception as e:
        return f"Error: Evaluation failed - {str(e)}"


def evaluate_quiz_expression(quiz_expr):
    """
    Process and evaluate a user-submitted quiz expression.
    
    Args:
        quiz_expr (str): The quiz expression to evaluate
        
    Returns:
        str: The result of the evaluation or 'Error!' if evaluation fails
    """
    if not isinstance(quiz_expr, str):
        return "Error!"
    
    # Use the safe_evaluate_expression function
    result = safe_evaluate_expression(quiz_expr)
    
    # Check if the result starts with "Error"
    if result.startswith("Error"):
        return "Error!"
    
    return result
