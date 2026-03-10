import re

def evaluate_expression(expression: str) -> str:
    """
    Evaluates a simple arithmetic expression string and returns the result as a string.
    Handles addition, subtraction, multiplication, and division.
    Returns 'Invalid Expression' for invalid or unsupported expressions.
    """
    # Validate allowed characters: numbers (integers/floats),
    # operators (+, -, *, /), parentheses, and whitespace.
    # This is a security measure to restrict input to `eval()`.
    # Using fullmatch to ensure the entire string conforms.
    allowed_chars_pattern = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]*$")
    if not allowed_chars_pattern.fullmatch(expression):
        return "Invalid Expression"

    # Remove leading/trailing whitespace.
    sanitized_expression = expression.strip()

    # If the expression is empty after stripping whitespace, it's invalid.
    if not sanitized_expression:
        return "Invalid Expression"

    try:
        # Using eval(). The regex and character validation above significantly
        # restrict what can be evaluated, aiming to confine it to arithmetic
        # operations as requested.
        result = eval(sanitized_expression)

        # Ensure the result is a numerical type (integer or float).
        # This handles cases where an expression might evaluate to something
        # non-numeric if the regex were less strict or if Python's eval
        # had surprising behavior for some validly-formed arithmetic string.
        if not isinstance(result, (int, float)):
            return "Invalid Expression"
            
        return str(result)
    
    # Order of exceptions: More specific ones first.
    except ZeroDivisionError:
        # Handles division by zero, e.g., "1 / 0" or "1 / (2 - 2)".
        return "Invalid Expression"
    except SyntaxError:
        # Handles malformed expressions, e.g., "1 +", "1 * * 2", "1.2.3".
        # eval() raises SyntaxError for these.
        return "Invalid Expression"
    except (TypeError, ValueError):
        # TypeError: e.g., an operation on incompatible types that might somehow pass regex.
        # ValueError: e.g., float("1.2.3a") if eval internals tried such a conversion.
        # For simple arithmetic with the current regex, these are less likely but good to cover.
        return "Invalid Expression"
    except OverflowError:
        # Handles cases where a calculation results in a number too large to be represented.
        # e.g., eval("1e308 * 10")
        return "Invalid Expression"
    except Exception:
        # A catch-all for any other unexpected errors during evaluation,
        # ensuring the function always returns one of the specified outputs.
        return "Invalid Expression"
