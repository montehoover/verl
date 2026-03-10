import re

def evaluate_math_expression(expression: str):
    """
    Evaluates a string containing a simple arithmetic expression.

    Args:
        expression: The string containing the arithmetic expression.
                    Allowed operations: +, -, *, /.
                    Allowed characters: digits, '.', '+', '-', '*', '/', '(', ')', whitespace.

    Returns:
        The calculated result if the expression is valid and safe,
        otherwise "Error!".
    """
    if not isinstance(expression, str):
        return "Error!"

    # Check for empty or whitespace-only strings, which are invalid expressions.
    if not expression.strip():
        return "Error!"

    # Validate allowed characters.
    # Allowed: digits (0-9), decimal point (.), operators (+, -, *, /),
    # parentheses (()), and whitespace (\s).
    # If any character not in this set is found, it's an error.
    forbidden_chars_pattern = r"[^0-9\.\+\-\*\/\(\)\s]"
    if re.search(forbidden_chars_pattern, expression):
        return "Error!"

    # Further check to prevent execution of arbitrary code, e.g. containing letters.
    # Although the pattern above should catch letters, this is an explicit safeguard.
    if re.search(r"[a-zA-Z_]", expression):
        return "Error!"

    try:
        # Evaluate the expression.
        # The character validation above significantly mitigates risks with eval().
        result = eval(expression)

        # Ensure the result is a number (integer or float).
        # This handles cases like eval("()") which results in a tuple.
        if not isinstance(result, (int, float)):
            return "Error!"

        return result
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, OverflowError):
        # Catch common errors during evaluation that indicate an invalid expression.
        return "Error!"
    except Exception:
        # Catch any other unexpected errors during evaluation.
        return "Error!"

if __name__ == '__main__':
    # Test cases
    print(f"'1 + 1': {evaluate_math_expression('1 + 1')}")  # Expected: 2
    print(f"'2 * 3 - 5 / 2': {evaluate_math_expression('2 * 3 - 5 / 2')}")  # Expected: 3.5
    print(f"'(1 + 2) * 3': {evaluate_math_expression('(1 + 2) * 3')}")  # Expected: 9
    print(f"'10 / 2.5': {evaluate_math_expression('10 / 2.5')}")  # Expected: 4.0
    print(f"'-5 + 2': {evaluate_math_expression('-5 + 2')}") # Expected: -3
    print(f"'1.0 + 2.0': {evaluate_math_expression('1.0 + 2.0')}") # Expected: 3.0

    # Invalid expressions
    print(f"'1 / 0': {evaluate_math_expression('1 / 0')}")  # Expected: Error!
    print(f"'1 + ': {evaluate_math_expression('1 + ')}")  # Expected: Error!
    print(f"'1 + foo': {evaluate_math_expression('1 + foo')}")  # Expected: Error!
    print(f"'import os': {evaluate_math_expression('import os')}")  # Expected: Error!
    print(f"""'__import__("os").system("echo hello")': {evaluate_math_expression('__import__("os").system("echo hello")')}""") # Expected: Error!
    print(f"Empty string '': {evaluate_math_expression('')}")  # Expected: Error!
    print(f"Whitespace string '   ': {evaluate_math_expression('   ')}")  # Expected: Error!
    print(f"'1.2.3 + 4': {evaluate_math_expression('1.2.3 + 4')}") # Expected: Error! (SyntaxError from eval)
    print(f"""'eval("1+1")': {evaluate_math_expression('eval("1+1")')}""") # Expected: Error!
    print(f"'(1+2)*3)': {evaluate_math_expression('(1+2)*3)')}") # Expected: Error! (SyntaxError from eval for unbalanced parentheses)
    print(f"'()': {evaluate_math_expression('()')}") # Expected: Error! (eval returns tuple, caught by isinstance check)
    print(f"True: {evaluate_math_expression(True)}") # Expected: Error! (input not a string)
