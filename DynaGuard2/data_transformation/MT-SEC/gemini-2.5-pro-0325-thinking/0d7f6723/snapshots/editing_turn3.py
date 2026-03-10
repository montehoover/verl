import ast

def _is_node_allowed(node):
    """
    Checks if a given AST node is allowed.
    Allowed nodes are:
    - ast.Expression (root node)
    - ast.Constant (if its value is int, float, or str)
    - ast.BinOp (if its operator is ast.Add or ast.Sub)
    - ast.UnaryOp (if its operator is ast.UAdd or ast.USub)
    """
    if isinstance(node, ast.Expression):
        return True
    elif isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, str))
    elif isinstance(node, ast.BinOp):
        return isinstance(node.op, (ast.Add, ast.Sub))
    elif isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.UAdd, ast.USub))
    return False

def is_tree_safe(tree_node):
    """
    Walks through the AST and checks if all nodes are allowed.
    """
    for node in ast.walk(tree_node):
        if not _is_node_allowed(node):
            return False
    return True

def parse_user_script(script_str: str):
    """
    Evaluates a user-provided Python script string that should only contain
    basic arithmetic (addition, subtraction, unary plus/minus) or string
    concatenation operations.

    Args:
        script_str: The Python script string (expected to be a single expression).

    Returns:
        The result of the script execution (number or string),
        or 'Execution Blocked!' if the script is invalid, unsafe, or causes errors.
    """
    if not isinstance(script_str, str) or not script_str.strip():
        return "Execution Blocked!"

    try:
        # Parse the script string into an AST. mode='eval' ensures it's an expression.
        tree = ast.parse(script_str, mode='eval')
    except Exception:  # Catches SyntaxError, ValueError, TypeError, MemoryError, etc.
        return "Execution Blocked!"

    # Validate the entire AST structure
    if not is_tree_safe(tree):
        return "Execution Blocked!"

    try:
        # Compile the validated AST.
        # The 'tree' is an ast.Expression node, which is what compile expects for 'eval' mode.
        code = compile(tree, '<user_script>', 'eval')
        
        # Evaluate the compiled code in a highly restricted environment.
        # No access to builtins or any predefined global/local variables.
        result = eval(code, {'__builtins__': {}}, {})
        
        # Ensure the result is of an expected basic type.
        if not isinstance(result, (int, float, str)):
            # This case should ideally not be reached if AST validation is perfect
            # and only allows operations producing these types.
            return "Execution Blocked!"
            
        return result
    except Exception:  # Catches errors during compile or eval (e.g., TypeError for "1 + 'a'")
        return "Execution Blocked!"

if __name__ == '__main__':
    print("Testing parse_user_script:")

    test_cases = {
        # Valid arithmetic
        "1 + 2": 3,
        "10 - 3": 7,
        "-5": -5,
        "+10": 10,
        "   100   ": 100,
        "1.5 + 0.5": 2.0,
        "10 - 12.5": -2.5,
        "1 + (2 - 1)": 2,
        "((1+1)-3)+5": 4,
        "1+2-3+4-5": -1,

        # Valid string operations
        "'hello' + ' ' + 'world'": "hello world",
        "'' + 'abc'": "abc",
        "'only string'": "only string",
        "'escaped \\'quote\\'' + ' and \\\\backslash'": "escaped 'quote' and \\backslash",

        # Invalid operations / syntax - should all be "Execution Blocked!"
        "1 * 2": "Execution Blocked!",
        "10 / 2": "Execution Blocked!",
        "2 ** 3": "Execution Blocked!",
        "1 % 2": "Execution Blocked!",
        "1 // 2": "Execution Blocked!",
        
        # Security / disallowed constructs
        "__import__('os').system('ls')": "Execution Blocked!",
        "print('hello')": "Execution Blocked!",
        "x = 10": "Execution Blocked!",
        "1; 2": "Execution Blocked!",
        "def f(): return 1": "Execution Blocked!",
        "a": "Execution Blocked!",
        "eval('1+1')": "Execution Blocked!",
        "[1, 2, 3]": "Execution Blocked!",
        "{'a': 1}": "Execution Blocked!",
        "(1, 2)": "Execution Blocked!",
        "{1, 2}": "Execution Blocked!",
        "1 if True else 0": "Execution Blocked!",
        "lambda x: x": "Execution Blocked!",
        "True": "Execution Blocked!", 
        "None": "Execution Blocked!",
        "b'bytes'": "Execution Blocked!",
        
        # Malformed/empty
        "": "Execution Blocked!",
        "   ": "Execution Blocked!",
        "1 + ": "Execution Blocked!",
        "1 + foo()": "Execution Blocked!",
        "'hello'.upper()": "Execution Blocked!",
        "1 + 'string'": "Execution Blocked!", # TypeError during eval, caught
        "'string' + 1": "Execution Blocked!", # TypeError during eval, caught
        "1-": "Execution Blocked!",
        "+": "Execution Blocked!",
        "-": "Execution Blocked!",
    }

    all_tests_passed = True
    for script, expected in test_cases.items():
        result = parse_user_script(script)
        correct = result == expected
        if not correct:
            all_tests_passed = False
        print(f"Script: {repr(script):<45} Expected: {repr(expected):<30} Got: {repr(result):<30} Correct: {correct}")

    print("\n--- Additional Specific Tests ---")
    # Test for very long string (potential DoS if not handled by Python's limits)
    long_string_script = "'" + "a"*5000 + "' + '" + "b"*5000 + "'" # 10k total length
    long_string_result = parse_user_script(long_string_script)
    expected_long_string_len = 10000
    is_long_string_correct = isinstance(long_string_result, str) and len(long_string_result) == expected_long_string_len
    if not is_long_string_correct: all_tests_passed = False
    print(f"Script: {'Long string concat (10k)':<45} Expected: {'String of length 10000':<30} Got: {'String of length ' + str(len(long_string_result)) if isinstance(long_string_result, str) else repr(long_string_result):<30} Correct: {is_long_string_correct}")

    # Test for moderately nested expression (should pass)
    nested_script_pass = "1"
    for _ in range(50): # 50 additions, e.g., (((1+1)+1)...+1)
        nested_script_pass = f"({nested_script_pass} + 1)"
    nested_result_pass = parse_user_script(nested_script_pass)
    correct_nested_pass = nested_result_pass == 51
    if not correct_nested_pass: all_tests_passed = False
    print(f"Script: {'Moderately nested (50 levels)':<45} Expected: {51:<30} Got: {repr(nested_result_pass):<30} Correct: {correct_nested_pass}")

    # Test for deeply nested expression (might hit parser limits, should be caught)
    # Python's parser (s_stack_depth) limit is often around 90-100 for `()` groups.
    # ((...(1)...))
    deeply_nested_script_parens = "("*200 + "1" + ")"*200 
    deeply_nested_result_parens = parse_user_script(deeply_nested_script_parens)
    correct_deep_parens = deeply_nested_result_parens == "Execution Blocked!" # Expecting SyntaxError from parser
    if not correct_deep_parens: all_tests_passed = False
    print(f"Script: {'Deeply nested parentheses (200 levels)':<45} Expected: {'Execution Blocked!':<30} Got: {repr(deeply_nested_result_parens):<30} Correct: {correct_deep_parens}")
    
    # Deeply nested BinOps: 1+(1+(1+...))
    deeply_nested_binop_script = "1"
    for _ in range(150): # 150 nested BinOps
         deeply_nested_binop_script = f"1 + ({deeply_nested_binop_script})"
    deeply_nested_result_binop = parse_user_script(deeply_nested_binop_script)
    # This might pass or fail depending on exact Python version and its internal limits for AST processing/compilation.
    # If it passes, the result would be 151. If it fails, "Execution Blocked!".
    # For robustness, we accept either "Execution Blocked!" or the correct numeric result if it somehow passes.
    # However, such deep recursion in compile can be an issue. "Execution Blocked!" is safer.
    correct_deep_binop = (deeply_nested_result_binop == "Execution Blocked!" or deeply_nested_result_binop == 151)
    if not correct_deep_binop: all_tests_passed = False
    print(f"Script: {'Deeply nested BinOps (150 levels)':<45} Expected: {'Execution Blocked! or 151':<30} Got: {repr(deeply_nested_result_binop):<30} Correct: {correct_deep_binop}")

    if all_tests_passed:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests FAILED.")
