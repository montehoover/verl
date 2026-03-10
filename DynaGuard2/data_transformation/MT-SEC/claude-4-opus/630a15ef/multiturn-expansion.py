import re

def is_safe_query(query):
    """
    Check if a string contains only numbers, basic arithmetic operators (+, -, *, /), and spaces.
    
    Args:
        query (str): The input string to validate
        
    Returns:
        bool: True if the input is safe, False otherwise
    """
    # Pattern matches only digits, arithmetic operators, spaces, and decimal points
    pattern = r'^[0-9+\-*/\s.]+$'
    
    if not query or not query.strip():
        return False
    
    return bool(re.match(pattern, query))


def evaluate_expression(expression):
    """
    Evaluate a mathematical expression with basic operators (+, -, *, /).
    
    Args:
        expression (str): The mathematical expression to evaluate
        
    Returns:
        float: The result of the expression
        
    Raises:
        ValueError: If the expression contains unsupported characters or is malformed
    """
    # First validate the expression
    if not is_safe_query(expression):
        raise ValueError("Expression contains unsupported characters")
    
    # Remove all spaces
    expression = expression.replace(' ', '')
    
    # Check for empty expression
    if not expression:
        raise ValueError("Empty expression")
    
    # Check for malformed expressions (consecutive operators, leading/trailing operators)
    if re.search(r'[+\-*/]{2,}', expression):
        raise ValueError("Malformed expression: consecutive operators")
    if re.match(r'^[*/]', expression) or re.search(r'[+\-*/]$', expression):
        raise ValueError("Malformed expression: invalid operator placement")
    
    try:
        # Use Python's built-in eval with restricted namespace for safety
        # Only allow basic math operations
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception:
        raise ValueError("Invalid expression")


def process_user_query(query):
    """
    Process a user-provided mathematical expression by validating and evaluating it.
    
    Args:
        query (str): The user-provided mathematical expression
        
    Returns:
        float: The computed result of the expression
        
    Raises:
        ValueError: If the input is unsafe or malformed
    """
    # Validate the query using is_safe_query
    if not is_safe_query(query):
        raise ValueError("Unsafe query: contains invalid characters")
    
    # Evaluate the expression using evaluate_expression
    return evaluate_expression(query)
