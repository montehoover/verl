import re

def evaluate_math_expression(expression: str):
    """
    Evaluates a string containing a simple arithmetic or basic string expression.

    Args:
        expression: The string containing the arithmetic or string expression.
                    Allowed operations: +, -, *, / for numbers;
                                      + (concatenation), * (repetition) for strings.
                    Allowed characters: digits, letters (for string content/literals),
                                      '.', '+', '-', '*', '/', '(', ')',
                                      whitespace, single and double quotes.

    Returns:
        The calculated result (number or string) if the expression is valid and safe,
        otherwise "Error!".
    """
    if not isinstance(expression, str):
        return "Error!"

    # Check for empty or whitespace-only strings, which are invalid expressions.
    if not expression.strip():
        return "Error!"

    # Validate allowed characters.
    # Allowed: digits (0-9), letters (a-zA-Z), decimal point (.),
    # operators (+, -, *, /), parentheses (()), whitespace (\s),
    # single quotes ('), and double quotes (").
    # If any character not in this set is found, it's an error.
    # This prevents use of characters like '_' which could lead to accessing
    # unintended attributes or methods (e.g., __import__).
    forbidden_chars_pattern = r"[^0-9a-zA-Z\.\+\-\*\/\(\)\s\'\"]"
    if re.search(forbidden_chars_pattern, expression):
        return "Error!"

    try:
        # Evaluate the expression.
        # The character validation above significantly mitigates risks with eval().
        result = eval(expression)

        # Ensure the result is a number (integer or float) or a string.
        # This handles cases like eval("()") which results in a tuple,
        # or other unexpected eval outcomes.
        if not isinstance(result, (int, float, str)):
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

    # String operations
    print(f"""'hello' + ' world': {evaluate_math_expression("'hello' + ' world'")}""") # Expected: hello world
    print(f"""\"abc\" + \"def\": {evaluate_math_expression('"abc" + "def"')}""") # Expected: abcdef
    print(f"""'abc' * 3: {evaluate_math_expression("'abc' * 3")}""") # Expected: abcabcabc
    print(f"""3 * 'abc': {evaluate_math_expression("3 * 'abc'")}""") # Expected: abcabcabc
    print(f"""('hello' + ' ') * 2 + 'world': {evaluate_math_expression("('hello' + ' ') * 2 + 'world'")}""") # Expected: hello hello world

    # Invalid expressions
    print(f"'1 / 0': {evaluate_math_expression('1 / 0')}")  # Expected: Error!
    print(f"'1 + ': {evaluate_math_expression('1 + ')}")  # Expected: Error!
    print(f"'1 + foo': {evaluate_math_expression('1 + foo')}")  # Expected: Error! (NameError for foo)
    expr_str_plus_int = "'\"hello\" + 1'"
    print(f"""'\\\"hello\\\" + 1': {evaluate_math_expression(expr_str_plus_int)}""") # Expected: Error! (TypeError for str + int)
    print(f"'import os': {evaluate_math_expression('import os')}")  # Expected: Error! (SyntaxError for import statement)
    print(f"""'__import__("os").system("echo hello")': {evaluate_math_expression('__import__("os").system("echo hello")')}""") # Expected: Error! (Forbidden char '_')
    print(f"Empty string '': {evaluate_math_expression('')}")  # Expected: Error!
    print(f"Whitespace string '   ': {evaluate_math_expression('   ')}")  # Expected: Error!
    print(f"'1.2.3 + 4': {evaluate_math_expression('1.2.3 + 4')}") # Expected: Error! (SyntaxError from eval)
    print(f"""'eval("1+1")': {evaluate_math_expression('eval("1+1")')}""") # Expected: Error!
    print(f"'(1+2)*3)': {evaluate_math_expression('(1+2)*3)')}") # Expected: Error! (SyntaxError from eval for unbalanced parentheses)
    print(f"'()': {evaluate_math_expression('()')}") # Expected: Error! (eval returns tuple, caught by isinstance check)
    print(f"True: {evaluate_math_expression(True)}") # Expected: Error! (input not a string)
