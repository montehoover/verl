import re
import ast

def validate_operation_input(input_string):
    """
    Validates if a string contains only numbers, basic arithmetic operators (+, -, *, /), and spaces.
    
    Args:
        input_string (str): The string to validate
        
    Returns:
        bool: True if the input is valid, False otherwise
    """
    # Pattern allows digits, operators (+, -, *, /), spaces, and decimal points
    pattern = r'^[0-9+\-*/.\s]+$'
    
    if not input_string or not input_string.strip():
        return False
    
    return bool(re.match(pattern, input_string))


def evaluate_expression(expression):
    """
    Evaluates a mathematical expression string and returns the result.
    
    Args:
        expression (str): The mathematical expression to evaluate
        
    Returns:
        float: The result of the evaluation
        
    Raises:
        ValueError: If the expression is invalid or cannot be evaluated
    """
    if not expression or not expression.strip():
        raise ValueError("Expression cannot be empty")
    
    # First validate the expression
    if not validate_operation_input(expression):
        raise ValueError("Expression contains invalid characters")
    
    try:
        # Check for division by zero before evaluation
        if '/0' in expression.replace(' ', ''):
            raise ValueError("Division by zero")
        
        # Evaluate the expression
        result = eval(expression)
        
        # Handle potential infinity or NaN results
        if result == float('inf') or result == float('-inf'):
            raise ValueError("Result is infinity")
        if result != result:  # NaN check
            raise ValueError("Result is not a number")
        
        return float(result)
    
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except (SyntaxError, NameError, TypeError) as e:
        raise ValueError(f"Invalid expression: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")


def safe_execute_operation(operation):
    """
    Safely executes a mathematical operation by validating and evaluating it.
    
    Args:
        operation (str): The mathematical operation to execute
        
    Returns:
        float: The result of the operation
        
    Raises:
        ValueError: If the operation is invalid or cannot be executed
    """
    # Validate the operation input
    if not validate_operation_input(operation):
        raise ValueError("Invalid operation: contains unauthorized characters")
    
    # Evaluate the expression
    return evaluate_expression(operation)
