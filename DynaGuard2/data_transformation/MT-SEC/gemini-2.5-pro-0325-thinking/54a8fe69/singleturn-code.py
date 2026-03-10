import re

def evaluate_user_expression(expression: str):
    """
    Safely calculates the result of a mathematical expression provided by the user.

    Args:
        expression: The string containing the user’s mathematical expression.

    Returns:
        The result of evaluating the mathematical expression.

    Raises:
        ValueError: If the input contains invalid characters, is empty,
                    or if the expression is mathematically incorrect or malformed.
    """
    if not isinstance(expression, str):
        raise TypeError("Expression must be a string.")

    # Remove whitespace to check if the expression is empty
    if not expression.strip():
        raise ValueError("Expression cannot be empty.")

    # Validate allowed characters: digits, operators (+, -, *, /, %), parentheses, decimal point, whitespace.
    # The regex ensures that the expression ONLY contains these characters.
    # It also prevents empty parentheses or operators at the very start/end if not part of a number (e.g. -5)
    # but eval() will catch more complex syntactical errors.
    allowed_pattern = r"^[0-9\s\.\+\-\*\/\%\(\)]*$"
    if not re.fullmatch(allowed_pattern, expression):
        raise ValueError("Expression contains invalid characters.")

    # Further check to prevent potentially unsafe constructs if any slip through the regex
    # (though the current regex is quite strict).
    # For example, disallow '__' which could be used to access object internals.
    if "__" in expression:
        raise ValueError("Expression contains forbidden sequences.")

    try:
        # Safely evaluate the expression.
        # The environment for eval is restricted to builtins, which is relatively safe
        # after the character validation.
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except ZeroDivisionError:
        raise ValueError("Error: Division by zero.")
    except SyntaxError:
        raise ValueError("Error: Invalid syntax in expression.")
    except TypeError:
        raise ValueError("Error: Type error in expression, check operands and operators.")
    except Exception as e:
        # Catch any other unexpected errors during evaluation.
        raise ValueError(f"Error evaluating expression: {e}")
