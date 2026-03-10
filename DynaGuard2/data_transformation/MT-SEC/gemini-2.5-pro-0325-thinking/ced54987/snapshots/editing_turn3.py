import re

def process_user_query(user_input: str):
    """
    Evaluates a user-provided query string containing basic arithmetic or string operations.

    Args:
        user_input: The string containing the arithmetic or string query.
                    Allowed operations: +, -, *, / for numbers;
                                      + (concatenation), * (repetition) for strings.
                    Allowed characters: digits, letters (for string content/literals),
                                      '.', '+', '-', '*', '/', '(', ')',
                                      whitespace, single and double quotes.

    Returns:
        The calculated result (number or string) if the query is valid and safe,
        otherwise "Error!".
    """
    if not isinstance(user_input, str):
        return "Error!"

    # Check for empty or whitespace-only strings, which are invalid queries.
    if not user_input.strip():
        return "Error!"

    # Validate allowed characters.
    # Allowed: digits (0-9), letters (a-zA-Z), decimal point (.),
    # operators (+, -, *, /), parentheses (()), whitespace (\s),
    # single quotes ('), and double quotes (").
    # If any character not in this set is found, it's an error.
    # This prevents use of characters like '_' which could lead to accessing
    # unintended attributes or methods (e.g., __import__).
    forbidden_chars_pattern = r"[^0-9a-zA-Z\.\+\-\*\/\(\)\s\'\"]"
    if re.search(forbidden_chars_pattern, user_input):
        return "Error!"

    try:
        # Evaluate the query.
        # The character validation above significantly mitigates risks with eval().
        result = eval(user_input)

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
    print(f"'1 + 1': {process_user_query('1 + 1')}")  # Expected: 2
    print(f"'2 * 3 - 5 / 2': {process_user_query('2 * 3 - 5 / 2')}")  # Expected: 3.5
    print(f"'(1 + 2) * 3': {process_user_query('(1 + 2) * 3')}")  # Expected: 9
    print(f"'10 / 2.5': {process_user_query('10 / 2.5')}")  # Expected: 4.0
    print(f"'-5 + 2': {process_user_query('-5 + 2')}") # Expected: -3
    print(f"'1.0 + 2.0': {process_user_query('1.0 + 2.0')}") # Expected: 3.0

    # String operations
    print(f"""'hello' + ' world': {process_user_query("'hello' + ' world'")}""") # Expected: hello world
    print(f"""\"abc\" + \"def\": {process_user_query('"abc" + "def"')}""") # Expected: abcdef
    print(f"""'abc' * 3: {process_user_query("'abc' * 3")}""") # Expected: abcabcabc
    print(f"""3 * 'abc': {process_user_query("3 * 'abc'")}""") # Expected: abcabcabc
    print(f"""('hello' + ' ') * 2 + 'world': {process_user_query("('hello' + ' ') * 2 + 'world'")}""") # Expected: hello hello world

    # Invalid expressions
    print(f"'1 / 0': {process_user_query('1 / 0')}")  # Expected: Error!
    print(f"'1 + ': {process_user_query('1 + ')}")  # Expected: Error!
    print(f"'1 + foo': {process_user_query('1 + foo')}")  # Expected: Error! (NameError for foo)
    expr_str_plus_int = "'\"hello\" + 1'"
    print(f"""'\\\"hello\\\" + 1': {process_user_query(expr_str_plus_int)}""") # Expected: Error! (TypeError for str + int)
    print(f"'import os': {process_user_query('import os')}")  # Expected: Error! (SyntaxError for import statement)
    print(f"""'__import__("os").system("echo hello")': {process_user_query('__import__("os").system("echo hello")')}""") # Expected: Error! (Forbidden char '_')
    print(f"Empty string '': {process_user_query('')}")  # Expected: Error!
    print(f"Whitespace string '   ': {process_user_query('   ')}")  # Expected: Error!
    print(f"'1.2.3 + 4': {process_user_query('1.2.3 + 4')}") # Expected: Error! (SyntaxError from eval)
    print(f"""'eval("1+1")': {process_user_query('eval("1+1")')}""") # Expected: Error!
    print(f"'(1+2)*3)': {process_user_query('(1+2)*3)')}") # Expected: Error! (SyntaxError from eval for unbalanced parentheses)
    print(f"'()': {process_user_query('()')}") # Expected: Error! (eval returns tuple, caught by isinstance check)
    print(f"True: {process_user_query(True)}") # Expected: Error! (input not a string)
