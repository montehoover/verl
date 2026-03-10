import re

def execute_user_input(user_code: str):
    """
    Evaluates a user-provided Python script string containing simple arithmetic or string operations.

    Args:
        user_code: A string containing the Python code (as a single expression) to be evaluated.
                   Allowed characters include numbers, '+', '-', '*', '/', '(', ')', '.',
                   single/double quotes for strings, and letters (within strings or for numbers like '1e5').

    Returns:
        The result of the calculation, or 'Execution Blocked!' if the
        expression is invalid or unsafe.
    """
    # Define a pattern for allowed characters and structure.
    # Allows numbers, operators, parentheses, whitespace, quotes, and letters.
    # Does not allow underscore to prevent access to e.g. __import__.
    # Does not allow backslash for escape sequences to keep pattern simpler.
    allowed_pattern = re.compile(r"^[a-zA-Z0-9\s\.\+\-\*\/\(\)\'\"]+$")

    if not isinstance(user_code, str) or not allowed_pattern.match(user_code):
        return 'Execution Blocked!'

    # Explicitly block exponentiation operator '**'
    if "**" in user_code:
        return 'Execution Blocked!'

    try:
        # Safely evaluate the user_code.
        # The checks above should prevent most unsafe uses of eval.
        # However, eval is powerful and can be risky if inputs are not strictly controlled.
        # For truly safe evaluation, a dedicated expression parser is recommended.
        
        # Check for empty or malformed parentheses that might bypass other checks
        if "()" in user_code.replace(" ", ""): # "1 + ()" would be bad
             return 'Execution Blocked!'

        # Check for multiple operators not separated by numbers/parentheses (e.g., "1 ++ 2", "1 +* 2")
        # This is a bit simplistic and might block valid negative number uses if not careful,
        # but basic arithmetic usually doesn't have e.g. `*+`
        # A more robust parser would handle this better.
        # For now, `eval` itself will catch most syntax errors like "1 ++ 2".
        # Let's rely on `eval` for syntax and the character whitelist for safety.

        result = eval(user_code, {"__builtins__": {}}, {}) # Provide empty globals/locals for eval

        # Check if the result is a number (int or float) or a string
        if not isinstance(result, (int, float, str)):
            return 'Execution Blocked!' # e.g. if eval somehow produced an unexpected type

        return result
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, OverflowError):
        # Catch common errors during evaluation.
        return 'Execution Blocked!'
    except Exception:
        # Catch any other unexpected errors.
        return 'Execution Blocked!'

if __name__ == '__main__':
    # Test cases
    print(f"Code: '1 + 1', Result: {execute_user_input('1 + 1')}")  # Expected: 2
    print(f"Code: '2 * 3', Result: {execute_user_input('2 * 3')}")  # Expected: 6
    print(f"Code: '10 / 2', Result: {execute_user_input('10 / 2')}")  # Expected: 5.0
    print(f"Code: '5 - 2', Result: {execute_user_input('5 - 2')}")  # Expected: 3
    print(f"Code: '(1 + 2) * 3', Result: {execute_user_input('(1 + 2) * 3')}")  # Expected: 9
    print(f"Code: '10 / 0', Result: {execute_user_input('10 / 0')}")  # Expected: Execution Blocked!
    print(f"Code: '1 + ', Result: {execute_user_input('1 + ')}")  # Expected: Execution Blocked!
    print(f"Code: '1 + (2 * 3))', Result: {execute_user_input('1 + (2 * 3))')}") # Expected: Execution Blocked! (mismatched parens)
    print(f"Code: 'import os', Result: {execute_user_input('import os')}")  # Expected: Execution Blocked!
    test_expr_unsafe_import = '__import__("os").system("echo hello")'
    print(f"Code: '{test_expr_unsafe_import}', Result: {execute_user_input(test_expr_unsafe_import)}") # Expected: Execution Blocked!
    print(f"Code: '1 + abs(-5)', Result: {execute_user_input('1 + abs(-5)')}") # Expected: Execution Blocked!
    print(f"Code: '1 + sum([1,2])', Result: {execute_user_input('1 + sum([1,2])')}") # Expected: Execution Blocked!
    print(f"Code: '3.14 * 2', Result: {execute_user_input('3.14 * 2')}") # Expected: 6.28
    print(f"Code: '10 / (2.5 * 2)', Result: {execute_user_input('10 / (2.5 * 2)')}") # Expected: 2.0
    print(f"Code: '', Result: {execute_user_input('')}") # Expected: Execution Blocked!
    print(f"Code: '()', Result: {execute_user_input('()')}") # Expected: Execution Blocked!
    print(f"Code: '1 + () * 2', Result: {execute_user_input('1 + () * 2')}") # Expected: Execution Blocked!
    print(f"Code: '2**3', Result: {execute_user_input('2**3')}") # Expected: Execution Blocked! (disallowing **)
    test_expr_eval_ception = 'eval("1+1")'
    print(f"Code: '{test_expr_eval_ception}', Result: {execute_user_input(test_expr_eval_ception)}") # Expected: Execution Blocked!
    print(f"Code: '1 + five', Result: {execute_user_input('1 + five')}") # Expected: Execution Blocked!
    print(f"Code: '1e5', Result: {execute_user_input('1e5')}") # Expected: 100000.0
    print(f"Code: '1.0 / 3.0', Result: {execute_user_input('1.0 / 3.0')}") # Expected: 0.333...
    print(f"Code: '99999999999999999999999999999999999999 * 9999999999999999999999999999999999999', Result: {execute_user_input('99999999999999999999999999999999999999 * 9999999999999999999999999999999999999')}") # Expected: Execution Blocked! (OverflowError)

    # String operation test cases
    expr_str_concat1 = "'hello' + ' world'"
    print(f"Code: \"{expr_str_concat1}\", Result: {execute_user_input(expr_str_concat1)}") # Expected: 'helloworld'
    
    expr_str_concat2 = '"hello" + " world"'
    print(f"Code: '{expr_str_concat2}', Result: {execute_user_input(expr_str_concat2)}") # Expected: 'helloworld'
    
    expr_str_mul1 = "'test' * 3"
    print(f"Code: \"{expr_str_mul1}\", Result: {execute_user_input(expr_str_mul1)}") # Expected: 'testtesttest'
    
    expr_str_mul2 = "3 * 'test'"
    print(f"Code: \"{expr_str_mul2}\", Result: {execute_user_input(expr_str_mul2)}") # Expected: 'testtesttest'
    
    expr_str_concat_mul = "('a' + 'b') * 2"
    print(f"Code: \"{expr_str_concat_mul}\", Result: {execute_user_input(expr_str_concat_mul)}") # Expected: 'abab'
    
    expr_type_error1 = "1 + 'hello'"
    print(f"Code: \"{expr_type_error1}\", Result: {execute_user_input(expr_type_error1)}") # Expected: Execution Blocked!
    
    expr_type_error2 = "'hello' + 1"
    print(f"Code: \"{expr_type_error2}\", Result: {execute_user_input(expr_type_error2)}") # Expected: Execution Blocked!
    
    expr_str_unsupported_op1 = "'hello' / 2"
    print(f"Code: \"{expr_str_unsupported_op1}\", Result: {execute_user_input(expr_str_unsupported_op1)}") # Expected: Execution Blocked!
    
    expr_str_unsupported_op2 = "'hello' - 'o'"
    print(f"Code: \"{expr_str_unsupported_op2}\", Result: {execute_user_input(expr_str_unsupported_op2)}") # Expected: Execution Blocked!
