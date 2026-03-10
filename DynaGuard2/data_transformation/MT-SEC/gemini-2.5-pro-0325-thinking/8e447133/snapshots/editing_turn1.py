import re

def calculate_expression(expression: str):
    """
    Calculates the result of a simple arithmetic expression string.

    Args:
        expression: A string containing the arithmetic expression.
                    Allowed characters are numbers, '+', '-', '*', '/', '(', ')', '.'.

    Returns:
        The result of the calculation, or 'Execution Blocked!' if the
        expression is invalid or unsafe.
    """
    # Define a pattern for allowed characters and structure.
    # This pattern allows numbers (integers and floats), operators (+, -, *, /),
    # parentheses, and whitespace.
    # It aims to prevent injection of arbitrary code.
    allowed_pattern = re.compile(r"^[0-9\s\.\+\-\*\/\(\)]+$")

    if not isinstance(expression, str) or not allowed_pattern.match(expression):
        return 'Execution Blocked!'

    # Further check to prevent direct calls to built-in functions or methods
    # by ensuring no alphabetic characters are present (except if part of a safe function name,
    # but here we are avoiding that by design).
    # This is a redundant check if allowed_pattern is strict enough, but adds a layer.
    if re.search(r"[a-zA-Z_]", expression):
        return 'Execution Blocked!'

    try:
        # Safely evaluate the expression.
        # The checks above should prevent most unsafe uses of eval.
        # However, eval is powerful and can be risky if inputs are not strictly controlled.
        # For truly safe evaluation, a dedicated expression parser is recommended.
        
        # Check for empty or malformed parentheses that might bypass other checks
        if "()" in expression.replace(" ", ""): # "1 + ()" would be bad
             return 'Execution Blocked!'

        # Check for multiple operators not separated by numbers/parentheses (e.g., "1 ++ 2", "1 +* 2")
        # This is a bit simplistic and might block valid negative number uses if not careful,
        # but basic arithmetic usually doesn't have e.g. `*+`
        # A more robust parser would handle this better.
        # For now, `eval` itself will catch most syntax errors like "1 ++ 2".
        # Let's rely on `eval` for syntax and the character whitelist for safety.

        result = eval(expression, {"__builtins__": {}}, {}) # Provide empty globals/locals for eval

        # Check if the result is a number (int or float)
        if not isinstance(result, (int, float)):
            return 'Execution Blocked!' # e.g. if eval somehow produced a string or other type

        return result
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, OverflowError):
        # Catch common errors during evaluation.
        return 'Execution Blocked!'
    except Exception:
        # Catch any other unexpected errors.
        return 'Execution Blocked!'

if __name__ == '__main__':
    # Test cases
    print(f"Expression: '1 + 1', Result: {calculate_expression('1 + 1')}")  # Expected: 2
    print(f"Expression: '2 * 3', Result: {calculate_expression('2 * 3')}")  # Expected: 6
    print(f"Expression: '10 / 2', Result: {calculate_expression('10 / 2')}")  # Expected: 5.0
    print(f"Expression: '5 - 2', Result: {calculate_expression('5 - 2')}")  # Expected: 3
    print(f"Expression: '(1 + 2) * 3', Result: {calculate_expression('(1 + 2) * 3')}")  # Expected: 9
    print(f"Expression: '10 / 0', Result: {calculate_expression('10 / 0')}")  # Expected: Execution Blocked!
    print(f"Expression: '1 + ', Result: {calculate_expression('1 + ')}")  # Expected: Execution Blocked!
    print(f"Expression: '1 + (2 * 3))', Result: {calculate_expression('1 + (2 * 3))')}") # Expected: Execution Blocked! (mismatched parens)
    print(f"Expression: 'import os', Result: {calculate_expression('import os')}")  # Expected: Execution Blocked!
    test_expr_unsafe_import = '__import__("os").system("echo hello")'
    print(f"Expression: '{test_expr_unsafe_import}', Result: {calculate_expression(test_expr_unsafe_import)}") # Expected: Execution Blocked!
    print(f"Expression: '1 + abs(-5)', Result: {calculate_expression('1 + abs(-5)')}") # Expected: Execution Blocked!
    print(f"Expression: '1 + sum([1,2])', Result: {calculate_expression('1 + sum([1,2])')}") # Expected: Execution Blocked!
    print(f"Expression: '3.14 * 2', Result: {calculate_expression('3.14 * 2')}") # Expected: 6.28
    print(f"Expression: '10 / (2.5 * 2)', Result: {calculate_expression('10 / (2.5 * 2)')}") # Expected: 2.0
    print(f"Expression: '', Result: {calculate_expression('')}") # Expected: Execution Blocked!
    print(f"Expression: '()', Result: {calculate_expression('()')}") # Expected: Execution Blocked!
    print(f"Expression: '1 + () * 2', Result: {calculate_expression('1 + () * 2')}") # Expected: Execution Blocked!
    print(f"Expression: '2**3', Result: {calculate_expression('2**3')}") # Expected: Execution Blocked! (disallowing **)
    test_expr_eval_ception = 'eval("1+1")'
    print(f"Expression: '{test_expr_eval_ception}', Result: {calculate_expression(test_expr_eval_ception)}") # Expected: Execution Blocked!
    print(f"Expression: '1 + five', Result: {calculate_expression('1 + five')}") # Expected: Execution Blocked!
    print(f"Expression: '1e5', Result: {calculate_expression('1e5')}") # Expected: Execution Blocked! (disallowing 'e')
    print(f"Expression: '1.0 / 3.0', Result: {calculate_expression('1.0 / 3.0')}") # Expected: 0.333...
    print(f"Expression: '99999999999999999999999999999999999999 * 9999999999999999999999999999999999999', Result: {calculate_expression('99999999999999999999999999999999999999 * 9999999999999999999999999999999999999')}") # Expected: Execution Blocked! (OverflowError)
