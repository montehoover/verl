import math

def calculate_expression(expression: str):
    """
    Calculates the natural logarithm of the result of a basic arithmetic expression string.
    Handles complex expressions with parentheses and nested operations.

    Args:
        expression: A string containing the arithmetic expression
                    (e.g., "(2 + 3) * (4 - 1)").

    Returns:
        The natural logarithm of the expression's result if valid and positive,
        otherwise "Invalid Expression!".
    """
    try:
        # For safety, in a real application, consider using ast.literal_eval
        # or a dedicated parsing library instead of eval() if the input
        # source is not fully trusted.
        # However, for basic arithmetic as requested, eval() is straightforward.
        
        # Check for allowed characters to prevent malicious input to some extent
        # Adding 'm', 'a', 't', 'h', 'l', 'o', 'g' for math.log if we were to allow it directly in expression
        # but the request is to evaluate the expression first, then take the log.
        allowed_chars = set("0123456789+-*/(). ")
        if not all(char in allowed_chars for char in expression):
            return "Invalid Expression!"

        # Ensure there's something to evaluate
        if not expression.strip():
            return "Invalid Expression!"

        # Evaluate the arithmetic expression
        value = eval(expression)

        # Calculate the natural logarithm of the result
        if value <= 0:
            # Logarithm is undefined for non-positive numbers
            return "Invalid Expression!"
        
        result = math.log(value)
        return result
    except (SyntaxError, NameError, TypeError, ZeroDivisionError, ValueError):
        # ValueError can be raised by math.log for non-positive inputs,
        # though we explicitly check for it above, it's good practice to include.
        return "Invalid Expression!"
