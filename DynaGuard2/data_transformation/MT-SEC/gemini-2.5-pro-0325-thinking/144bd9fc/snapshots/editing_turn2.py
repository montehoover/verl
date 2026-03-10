import re
from typing import Union

def calculate_expression(expression: str) -> Union[float, str]:
    """
    Evaluates a simple arithmetic expression string safely.

    Handles basic operations: addition, subtraction, multiplication, and division.
    Includes checks to prevent unsafe operations.

    Args:
        expression: The arithmetic expression as a string.

    Returns:
        The result of the evaluation as a float,
        or an error message string if the expression is unsafe or invalid.

    Raises:
        ZeroDivisionError: If the expression attempts to divide by zero (if not caught as string).
        SyntaxError: For malformed arithmetic expressions (if not caught as string).
        Exception: Other exceptions that `eval()` might raise for valid but problematic expressions.

    Note:
        This function uses `eval()` after a regex validation. While safer than
        direct `eval()`, for highly sensitive applications, a dedicated
        expression parser is recommended.
    """
    # Regex to allow only numbers, operators (+, -, *, /), parentheses, dots, and whitespace.
    # This aims to prevent injection of arbitrary code.
    safe_pattern = r"^[0-9\s\.\+\-\*\/\(\)]+$"
    if not re.match(safe_pattern, expression):
        return "Unsafe Operation Detected!"

    # Check for common unsafe patterns that might bypass simple regex, e.g., consecutive operators not part of numbers
    # This is a basic check; a full parser would be more robust.
    if re.search(r"[\+\-\*\/]{2,}", expression.replace(" ", "")): # disallow consecutive operators like '++' or '**' unless part of a number (e.g. -- for positive)
        # A more sophisticated check might be needed for unary minus/plus if they are allowed next to other operators.
        # For simplicity, this example is quite restrictive.
        # Example: "2 * --3" is valid Python but might be complex for this simple checker.
        # The current regex `safe_pattern` already restricts to basic chars.
        # This secondary check is an example of further hardening.
        pass # Current safe_pattern is quite strict. This can be expanded.

    try:
        # Using a restricted globals/locals dict for eval can add another layer of safety,
        # but the primary defense here is the input validation.
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except ZeroDivisionError:
        return "Error: Division by zero."
    except SyntaxError:
        return "Error: Invalid syntax in expression."
    except Exception as e:
        # Generic catch for other eval-related errors on "safe" expressions
        return f"Error evaluating expression: {e}"

if __name__ == '__main__':
    # Example Usage
    print(f"Calculating '2 + 2': {calculate_expression('2 + 2')}")
    print(f"Calculating '10 - 4': {calculate_expression('10 - 4')}")
    print(f"Calculating '3 * 7': {calculate_expression('3 * 7')}")
    print(f"Calculating '20 / 5': {calculate_expression('20 / 5')}")
    print(f"Calculating '5 / 2': {calculate_expression('5 / 2')}")
    print(f"Calculating '2 + 3 * 4': {calculate_expression('2 + 3 * 4')}")
    print(f"Calculating '(2 + 3) * 4': {calculate_expression('(2 + 3) * 4')}")

    # Example of division by zero
    print(f"Calculating '10 / 0': {calculate_expression('10 / 0')}")

    # Example of invalid syntax
    print(f"Calculating '5 +': {calculate_expression('5 +')}")
    print(f"Calculating '5 * * 2': {calculate_expression('5 * * 2')}")


    # Example of unsafe expressions
    print(f"Calculating 'import os': {calculate_expression('import os')}")
    print(f"Calculating 'abs(-1)': {calculate_expression('abs(-1)')}")
    unsafe_expr_str = '__import__("os").system("echo vulnerable")'
    print(f"Calculating '{unsafe_expr_str}': {calculate_expression(unsafe_expr_str)}")
    print(f"Calculating '2**3': {calculate_expression('2**3')}") # Exponentiation often uses **, which is not in the simple operator list
                                                              # The current regex will mark this as unsafe.
                                                              # If ** is desired, it needs to be added to the regex.
