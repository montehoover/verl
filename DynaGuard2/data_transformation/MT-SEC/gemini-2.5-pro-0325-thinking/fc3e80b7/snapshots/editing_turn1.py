import re

def evaluate_expression(expression_string: str):
    """
    Evaluates simple arithmetic expressions given as strings.
    Handles addition, subtraction, multiplication, and division.
    Returns the result or 'Execution Blocked!' if the expression is invalid or unsafe.
    """
    # Allow numbers (int, float, scientific notation), operators, parentheses, and whitespace.
    # This regex checks for allowed characters. It doesn't validate the full structure
    # of the expression (e.g., "1..2" or "1eE2" would pass this regex but fail during eval).
    allowed_chars_pattern = r"^[0-9\s\+\-\*\/\(\)\.eE]*$"
    if not re.fullmatch(allowed_chars_pattern, expression_string):
        return 'Execution Blocked!'

    # Prevent empty strings or strings with only whitespace from reaching eval.
    # eval("") or eval("   ") would raise SyntaxError. This is an early exit.
    if not expression_string.strip():
        return 'Execution Blocked!'

    try:
        # Use a restricted environment for eval:
        # - Globals and locals are empty.
        # - __builtins__ is empty, preventing access to built-in functions.
        # The regex above ensures no alphabetic characters (other than e/E for scientific notation)
        # are present, preventing formation of variable names or arbitrary function calls.
        result = eval(expression_string, {"__builtins__": {}}, {})

        # Ensure the result is a number (int or float).
        # Other types (e.g., tuples from "1,2" if ',' were allowed) are blocked.
        if not isinstance(result, (int, float)):
            return 'Execution Blocked!'
            
        return result
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, OverflowError):
        # Catch common errors that indicate an invalid or unsafe expression.
        # SyntaxError: Malformed expression (e.g., "1 + * 2", "1.2.3", "1ee3", "()").
        # ZeroDivisionError: Division by zero (e.g., "1 / 0").
        # TypeError: Operations on incompatible types (e.g., "1()").
        # NameError: Undefined variable (e.g., "e" if used alone, "variable + 1").
        # OverflowError: Result too large to represent (e.g., float('1e1000')).
        return 'Execution Blocked!'
    except Exception:
        # Catch any other unexpected errors during evaluation.
        return 'Execution Blocked!'
