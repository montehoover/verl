def calculate_expression(expression: str):
    """
    Calculates the result of a basic arithmetic expression string.

    Args:
        expression: A string containing the arithmetic expression
                    (e.g., "1 + 1", "5 * 2 - 3 / 4").

    Returns:
        The result of the expression if valid, otherwise "Invalid Expression!".
    """
    try:
        # For safety, in a real application, consider using ast.literal_eval
        # or a dedicated parsing library instead of eval() if the input
        # source is not fully trusted.
        # However, for basic arithmetic as requested, eval() is straightforward.
        
        # Check for allowed characters to prevent malicious input to some extent
        allowed_chars = set("0123456789+-*/(). ")
        if not all(char in allowed_chars for char in expression):
            return "Invalid Expression!"

        # Ensure there's something to evaluate
        if not expression.strip():
            return "Invalid Expression!"

        result = eval(expression)
        return result
    except (SyntaxError, NameError, TypeError, ZeroDivisionError):
        return "Invalid Expression!"
