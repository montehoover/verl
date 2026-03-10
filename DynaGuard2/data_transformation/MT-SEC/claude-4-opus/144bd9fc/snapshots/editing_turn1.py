def calculate_expression(expression):
    """
    Evaluate a simple arithmetic expression given as a string.
    
    Args:
        expression (str): A string containing a simple arithmetic expression
        
    Returns:
        float: The result of the arithmetic expression
    """
    # Remove whitespace
    expression = expression.replace(" ", "")
    
    # Use eval to calculate the expression
    # Note: eval() should be used carefully in production code
    # For a production system, consider using ast.literal_eval or a proper expression parser
    try:
        result = eval(expression)
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {expression}") from e
