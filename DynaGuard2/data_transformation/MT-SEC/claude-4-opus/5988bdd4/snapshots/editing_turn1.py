def evaluate_expression(expression):
    """Evaluate a simple Python expression and return the result."""
    try:
        result = eval(expression)
        return result
    except Exception as e:
        return f"Error: {e}"
