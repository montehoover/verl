def evaluate_user_math(math_expr: str):
    """
    Evaluates a string containing a basic arithmetic expression.

    Args:
        math_expr: The string expression to evaluate (e.g., "2 + 2", "10 / 2").

    Returns:
        The result of the computation or "Processing Failed!" if an error occurs.
    """
    try:
        # For safety, it's better to use a dedicated math expression parser
        # for untrusted input. eval() can be dangerous with arbitrary strings.
        # However, for basic arithmetic as requested, and assuming controlled input,
        # it can be used.
        # We restrict the available names to a safe subset if possible,
        # or ensure the input string format is strictly validated.
        # For this example, we'll proceed with direct eval, but with a broad exception handler.
        
        # Allow numbers, arithmetic operators, parentheses, space, and letters for keywords (True, False, and, or, not)
        allowed_chars = set("0123456789+-*/(). abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if not all(char in allowed_chars for char in math_expr):
            return "Processing Failed!"

        # Define a safe environment for eval, explicitly including True and False
        eval_globals = {
            "__builtins__": {},
            "True": True,
            "False": False
        }
        # Note: 'and', 'or', 'not' are Python keywords and handled by eval if syntax is correct.
        
        result = eval(math_expr, eval_globals, {}) # Restrict access to builtins
        return result
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, OverflowError):
        return "Processing Failed!"
    except Exception:
        # Catch any other unexpected errors during evaluation
        return "Processing Failed!"

if __name__ == '__main__':
    # Test cases
    print(f"'3 + 5': {evaluate_user_math('3 + 5')}")  # Expected: 8
    print(f"'10 - 2': {evaluate_user_math('10 - 2')}")  # Expected: 8
    print(f"'4 * 6': {evaluate_user_math('4 * 6')}")  # Expected: 24
    print(f"'8 / 2': {evaluate_user_math('8 / 2')}")  # Expected: 4.0
    print(f"'5 / 0': {evaluate_user_math('5 / 0')}")  # Expected: Processing Failed!
    print(f"'2 + ': {evaluate_user_math('2 + ')}")  # Expected: Processing Failed!
    print(f"'hello + world': {evaluate_user_math('hello + world')}") # Expected: Processing Failed!
    hacked_expr_display_str = "'__import__(\"os\").system(\"echo hacked\")'"
    hacked_expr_eval_str = '__import__("os").system("echo hacked")'
    print(f"{hacked_expr_display_str}: {evaluate_user_math(hacked_expr_eval_str)}") # Expected: Processing Failed!
    print(f"'(2+3)*4': {evaluate_user_math('(2+3)*4')}") # Expected: 20
    print(f"'100 / (2 * 5)': {evaluate_user_math('100 / (2 * 5)')}") # Expected: 10.0
    print(f"'100 / 2 * 5': {evaluate_user_math('100 / 2 * 5')}") # Expected: 250.0

    # Logical operation test cases
    print(f"'True and False': {evaluate_user_math('True and False')}")  # Expected: False
    print(f"'True or False': {evaluate_user_math('True or False')}")  # Expected: True
    print(f"'not True': {evaluate_user_math('not True')}")  # Expected: False
    print(f"'not False': {evaluate_user_math('not False')}")  # Expected: True
    print(f"'(True and True) or False': {evaluate_user_math('(True and True) or False')}") # Expected: True
    print(f"'not (False or False)': {evaluate_user_math('not (False or False)')}") # Expected: True
    print(f"'True and (False or True)': {evaluate_user_math('True and (False or True)')}") # Expected: True
    # Test with mixed case keywords (should work as letters are allowed, and eval handles Python keywords case-sensitively)
    print(f"'true and false': {evaluate_user_math('true and false')}")  # Expected: Processing Failed! (NameError for 'true')
    print(f"'True AND False': {evaluate_user_math('True AND False')}")  # Expected: Processing Failed! (NameError for 'AND')
    # Test invalid expression
    print(f"'True and or False': {evaluate_user_math('True and or False')}") # Expected: Processing Failed! (SyntaxError)
