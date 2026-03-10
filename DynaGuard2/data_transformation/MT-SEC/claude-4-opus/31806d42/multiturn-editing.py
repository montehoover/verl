import re
import ast

def evaluate_expression(expr):
    """
    Evaluate a mathematical expression given as a string.
    
    Args:
        expr (str): A string representing the arithmetic expression
    
    Returns:
        float or int: The computed result of the expression
    
    Raises:
        ValueError: If the expression contains unsupported characters or unsafe operations
    """
    # Remove whitespace
    expr = expr.strip()
    
    # Check for empty expression
    if not expr:
        raise ValueError("Empty expression")
    
    # Validate that expression only contains allowed characters
    allowed_pattern = r'^[0-9\+\-\*/\(\)\.\s]+$'
    if not re.match(allowed_pattern, expr):
        raise ValueError("Expression contains unsupported characters")
    
    try:
        # Parse the expression into an AST
        node = ast.parse(expr, mode='eval')
        
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
            ast.Constant,  # Python 3.8+
            ast.Num,       # For older Python versions
        }
        
        # Check if all nodes in the AST are allowed
        for node in ast.walk(node):
            if type(node) not in allowed_nodes:
                # Also check for Name nodes with numeric values (for compatibility)
                if isinstance(node, ast.Name):
                    raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
        
        # Evaluate the expression
        result = eval(compile(node, '<string>', 'eval'))
        
        # Convert to float for consistency
        return float(result)
        
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")

def basic_calculator(num1, num2=None, operation=None):
    """
    Perform basic arithmetic operations on two numbers or parse and evaluate a string expression.
    
    Args:
        num1 (float or str): First number or string expression like '3 + 4'
        num2 (float, optional): Second number (not used if num1 is a string)
        operation (str, optional): Operation to perform ('+', '-', '*', '/') (not used if num1 is a string)
    
    Returns:
        float: Result of the operation
    """
    # If num1 is a string, parse it as an expression
    if isinstance(num1, str):
        # Regular expression to match number operator number
        pattern = r'^\s*(-?\d+\.?\d*)\s*([\+\-\*/])\s*(-?\d+\.?\d*)\s*$'
        match = re.match(pattern, num1)
        
        if not match:
            raise ValueError(f"Invalid expression format: {num1}")
        
        operand1 = float(match.group(1))
        operator = match.group(2)
        operand2 = float(match.group(3))
        
        return basic_calculator(operand1, operand2, operator)
    
    # Original functionality for separate arguments
    if operation == '+':
        return num1 + num2
    elif operation == '-':
        return num1 - num2
    elif operation == '*':
        return num1 * num2
    elif operation == '/':
        if num2 == 0:
            raise ValueError("Division by zero is not allowed")
        return num1 / num2
    else:
        raise ValueError(f"Invalid operation: {operation}. Use '+', '-', '*', or '/'")
