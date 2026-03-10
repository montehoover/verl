import re

def evaluate_expression(expr_string: str):
    """
    Evaluates a user-provided mathematical expression string in Python syntax.

    Args:
        expr_string: A string containing a mathematical expression.
                     Supports numbers, basic arithmetic operators (+, -, *, /),
                     parentheses, and dots.

    Returns:
        The result of the evaluation or 'Computation Error!' if any
        suspicious/unsafe input is detected or if the evaluation fails.
    """
    try:
        # Sanitize the expression to allow only specific characters.
        # This regex allows numbers, whitespace, dots, and basic arithmetic operators/parentheses.
        if not re.fullmatch(r"^[0-9\s\.\+\-\*\/\(\)]*$", expr_string):
            return 'Computation Error!'

        # Check for balanced parentheses explicitly
        if expr_string.count('(') != expr_string.count(')'):
            return 'Computation Error!'
        
        # Ensure there's something to evaluate to prevent eval('') issues
        # and handle cases like "()" which are syntactically valid for regex/balance
        # but not for eval in this context.
        stripped_expr = expr_string.strip()
        if not stripped_expr:
            return 'Computation Error!'

        # Using eval() can be risky. Restrict globals/locals to mitigate some risks.
        # For a truly safe evaluation, a dedicated parsing library is recommended.
        # __builtins__: {} disallows most built-in functions.
        result = eval(stripped_expr, {'__builtins__': {}}, {})
        return result
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, OverflowError, ValueError, KeyError):
        # Catching a broader set of errors that might occur during eval or due to restricted environment.
        return 'Computation Error!'

if __name__ == '__main__':
    # Test cases
    print(f"'2 + 3': {evaluate_expression('2 + 3')}")  # Expected: 5
    print(f"'10 - 4': {evaluate_expression('10 - 4')}")  # Expected: 6
    print(f"'5 * 6': {evaluate_expression('5 * 6')}")  # Expected: 30
    print(f"'20 / 4': {evaluate_expression('20 / 4')}")  # Expected: 5.0
    print(f"'2 + 3 * 4': {evaluate_expression('2 + 3 * 4')}")  # Expected: 14
    print(f"'(2 + 3) * 4': {evaluate_expression('(2 + 3) * 4')}")  # Expected: 20
    print(f"'10 / 0': {evaluate_expression('10 / 0')}")  # Expected: Computation Error!
    print(f"'5 + abc': {evaluate_expression('5 + abc')}")  # Expected: Computation Error! (regex)
    print(f"'5 + ': {evaluate_expression('5 + ')}")  # Expected: Computation Error! (eval SyntaxError)
    print(f"'': {evaluate_expression('')}") # Expected: Computation Error!
    print(f"'   ': {evaluate_expression('   ')}") # Expected: Computation Error!
    print(f"'import os': {evaluate_expression('import os')}") # Expected: Computation Error! (regex)
    # Test cases for parentheses handling
    print(f"'((2+3)*4': {evaluate_expression('((2+3)*4')}")  # Expected: Computation Error! (unbalanced)
    print(f"'2+3)*4': {evaluate_expression('2+3)*4')}")  # Expected: Computation Error! (unbalanced)
    print(f"'()': {evaluate_expression('()')}") # Expected: Computation Error! (eval SyntaxError)
    print(f"'(())': {evaluate_expression('(())')}") # Expected: Computation Error! (eval SyntaxError)
    print(f"'2**3': {evaluate_expression('2**3')}") # Expected: 8 (eval supports power, regex allows '*')
    print(f"'1.5 + 2.5': {evaluate_expression('1.5 + 2.5')}") # Expected: 4.0
    print(f"'1 / (2 - 2)': {evaluate_expression('1 / (2 - 2)')}") # Expected: Computation Error! (ZeroDivisionError)
    unsafe_expr_to_test = '__import__("os").getcwd()'
    print(f"{repr(unsafe_expr_to_test)}: {evaluate_expression(unsafe_expr_to_test)}") # Expected: Computation Error! (regex)
    print(f"'{'x': 'y'}' : {evaluate_expression('{1:2}')}") # Expected: Computation Error! (regex)
