import re

def safe_eval_expression(expr: str):
    """
    Safely calculates the result of a mathematical expression provided by the user.

    Args:
        expr: str, the string containing the user’s mathematical expression.

    Returns:
        The result of evaluating the mathematical expression.

    Raises:
        ValueError: if the input contains invalid characters or if the expression is incorrect.
    """
    # Allow numbers, operators (+, -, *, /), parentheses, and whitespace.
    # Disallow any letters or other symbols to prevent arbitrary code execution.
    # The pattern checks if the entire string consists of allowed characters.
    # It allows for floating point numbers as well.
    allowed_pattern = r"^[0-9\s\.\+\-\*\/\(\)]+$"

    if not re.match(allowed_pattern, expr):
        raise ValueError("Invalid characters in expression.")

    # Further check for potentially unsafe patterns, e.g., consecutive operators not allowed by eval
    # or empty parentheses, though eval might catch some of these as SyntaxError.
    # This is a simple check; more complex validation might be needed for production systems.
    if re.search(r"[\+\-\*\/]{2,}", expr.replace(" ", "")): # Avoids things like "5++3"
        raise ValueError("Invalid expression format: consecutive operators.")
    if re.search(r"\(\s*\)", expr): # Avoids "()"
        raise ValueError("Invalid expression format: empty parentheses.")

    try:
        # Using eval() can be risky if the input is not strictly controlled.
        # The regex above is a safeguard, but for truly secure applications,
        # consider using a dedicated math expression parser library.
        result = eval(expr)
        return result
    except (SyntaxError, TypeError, ZeroDivisionError) as e:
        raise ValueError(f"Incorrect expression: {e}")
    except Exception as e:
        # Catch any other unexpected errors during evaluation
        raise ValueError(f"Error evaluating expression: {e}")

if __name__ == '__main__':
    # Example Usage:
    test_expressions = {
        "5 * 9": 45,
        "10 + 2 * 6": 22,
        "(10 + 2) * 6": 72,
        "100 / 4": 25,
        "2.5 * 4": 10.0,
        "  5   *  ( 3 + 1 ) ": 20
    }

    for expr_str, expected in test_expressions.items():
        try:
            output = safe_eval_expression(expr_str)
            print(f"Input: \"{expr_str}\", Output: {output}, Expected: {expected}, Match: {output == expected}")
        except ValueError as e:
            print(f"Input: \"{expr_str}\", Error: {e}")

    print("\nTesting invalid expressions:")
    invalid_expressions = [
        "5 * nine",
        "import os",
        "5 ** 2", # Exponentiation not explicitly allowed by current regex/logic
        "5 + ",
        "()",
        "5 ++ 3",
        "10 / 0"
    ]
    for expr_str in invalid_expressions:
        try:
            output = safe_eval_expression(expr_str)
            print(f"Input: \"{expr_str}\", Output: {output} (Unexpected success)")
        except ValueError as e:
            print(f"Input: \"{expr_str}\", Error: {e} (Expected)")
