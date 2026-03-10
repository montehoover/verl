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
        
        allowed_chars = set("0123456789+-*/(). ")
        if not all(char in allowed_chars for char in math_expr):
            return "Processing Failed!"

        result = eval(math_expr, {"__builtins__": {}}, {}) # Restrict access to builtins
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
