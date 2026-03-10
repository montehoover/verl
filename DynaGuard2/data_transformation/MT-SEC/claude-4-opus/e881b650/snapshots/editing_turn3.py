import re
import ast

def compute_expression(expr):
    """
    Evaluate a mathematical expression from a string.
    
    Args:
        expr (str): Mathematical expression to evaluate
    
    Returns:
        float or int: Result of the expression
    
    Raises:
        ValueError: If expression contains unsupported characters or can't be evaluated safely
    """
    # Remove whitespace
    expr = expr.strip()
    
    # Check for empty expression
    if not expr:
        raise ValueError("Empty expression")
    
    # Only allow digits, operators, parentheses, and decimal points
    if not re.match(r'^[0-9+\-*/().\s]+$', expr):
        raise ValueError("Expression contains unsupported characters")
    
    try:
        # Parse the expression into an AST
        node = ast.parse(expr, mode='eval')
        
        # Validate that only safe operations are used
        for n in ast.walk(node):
            if not isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, 
                                ast.Add, ast.Sub, ast.Mult, ast.Div,
                                ast.USub, ast.UAdd, ast.Num, ast.Constant)):
                raise ValueError("Unsupported operation in expression")
        
        # Evaluate the expression
        result = eval(compile(node, '<string>', 'eval'))
        
        # Handle division by zero
        if isinstance(result, float) and (result == float('inf') or result == float('-inf')):
            raise ValueError("Division by zero")
        
        return result
        
    except (SyntaxError, TypeError):
        raise ValueError("Invalid expression syntax")
    except ZeroDivisionError:
        raise ValueError("Division by zero")

def basic_calculate(num1, num2=None, operator=None):
    """
    Perform basic arithmetic operations on two numbers or parse a string expression.
    
    Args:
        num1: Either a float for the first number, or a string expression like '4 + 5'
        num2 (float, optional): Second number (not used if num1 is a string)
        operator (str, optional): Operation to perform ('+', '-', '*', '/') (not used if num1 is a string)
    
    Returns:
        float: Result of the operation
    """
    # If num1 is a string, parse it
    if isinstance(num1, str):
        # Split the expression
        parts = num1.strip().split()
        if len(parts) != 3:
            raise ValueError("Expression must be in format 'number operator number'")
        
        try:
            parsed_num1 = float(parts[0])
            parsed_operator = parts[1]
            parsed_num2 = float(parts[2])
        except (ValueError, IndexError):
            raise ValueError("Invalid expression format")
        
        return basic_calculate(parsed_num1, parsed_num2, parsed_operator)
    
    # Original functionality
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ValueError("Cannot divide by zero")
        return num1 / num2
    else:
        raise ValueError(f"Invalid operator: {operator}")
