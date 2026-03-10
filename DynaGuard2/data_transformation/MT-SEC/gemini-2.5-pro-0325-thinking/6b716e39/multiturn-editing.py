from asteval import Interpreter

def evaluate_user_math(math_expr: str) -> str:
    """
    Evaluates a string containing arithmetic or logical expressions safely using asteval.

    Args:
        math_expr: The string expression to evaluate.

    Returns:
        The result of the computation as a string, or "Processing Failed!" if an error occurs.
    """
    aeval = Interpreter()  # By default, uses a restricted symbol table and raises errors.
    try:
        result = aeval.eval(math_expr)
        # asteval.Interpreter raises errors by default for malformed/unsafe expressions.
        # The result from aeval.eval() is the actual computed value (e.g., int, float, bool).
        return str(result)
    except Exception:
        # This catches errors raised by asteval (e.g., SyntaxError, NameError, TypeError,
        # ZeroDivisionError, issues with node_evaluators for unsupported AST nodes)
        # or any other unexpected issue during evaluation.
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
