import ast

def _is_allowed_node(node):
    """
    Recursively checks if an AST node is allowed for basic arithmetic.
    Allowed nodes: Constants (numbers), binary operations (+, -, *, /, //, %, **),
                   unary operations (+, -).
    """
    if isinstance(node, ast.Expression):
        # The top-level node in 'eval' mode is Expression, check its body
        return _is_allowed_node(node.body)
    elif isinstance(node, ast.Constant):
        # Allow numeric constants (int, float, complex)
        return isinstance(node.value, (int, float, complex))
    elif isinstance(node, ast.BinOp):
        # Check if the binary operator is one of the allowed arithmetic ones
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, 
                                     ast.FloorDiv, ast.Mod, ast.Pow)):
            return False
        # Recursively check left and right operands
        return _is_allowed_node(node.left) and _is_allowed_node(node.right)
    elif isinstance(node, ast.UnaryOp):
        # Check if the unary operator is one of the allowed arithmetic ones
        if not isinstance(node.op, (ast.UAdd, ast.USub)): # Unary plus, Unary minus
            return False
        # Recursively check the operand
        return _is_allowed_node(node.operand)
    
    # If the node type is not explicitly allowed, reject it.
    # This implicitly disallows ast.Name (variables), ast.Call (function calls),
    # ast.Attribute, ast.Subscript, and any other potentially harmful constructs.
    return False

def parse_user_expression(expression_string: str) -> bool:
    """
    Parses a user input string to determine if it's a valid Python expression
    for basic arithmetic operations and doesn't contain harmful commands.

    Args:
        expression_string: The string input from the user.

    Returns:
        True if the input is a valid and safe arithmetic expression, False otherwise.
    """
    if not isinstance(expression_string, str) or not expression_string.strip():
        # Reject non-string or empty/whitespace-only input
        return False

    try:
        # Parse the expression string into an Abstract Syntax Tree (AST).
        # mode='eval' is used because we expect a single expression.
        parsed_ast = ast.parse(expression_string, mode='eval')
    except SyntaxError:
        # If ast.parse raises a SyntaxError, the string is not valid Python syntax.
        return False
    except Exception:
        # Catch any other parsing-related errors (e.g., recursion depth, memory errors for huge inputs)
        # For security, treat any unexpected parsing error as invalid.
        return False

    # Validate the entire AST structure to ensure it only contains allowed nodes/operations.
    return _is_allowed_node(parsed_ast)


def evaluate_safe_expression(expression_string: str):
    """
    Safely evaluates a validated Python arithmetic expression string.

    Args:
        expression_string: The string input from the user, expected to be
                           a basic arithmetic expression.

    Returns:
        The result of the evaluation if successful and safe,
        or an error message string if validation fails, evaluation fails,
        or an unsafe operation is attempted.
    """
    if not parse_user_expression(expression_string):
        return "Error: Invalid or unsafe expression."

    try:
        # Safely evaluate the expression.
        # The __builtins__ dictionary is restricted to prevent access to
        # harmful built-in functions. Only 'None' is typically needed for
        # simple expression evaluation, but an empty dict is safer.
        # The globals dictionary is also restricted.
        result = eval(expression_string, {"__builtins__": {}}, {})
        return result
    except ZeroDivisionError:
        return "Error: Division by zero."
    except OverflowError:
        return "Error: Numerical result out of range."
    except Exception as e:
        # Catch any other unexpected errors during evaluation.
        return f"Error: Evaluation failed ({type(e).__name__})."


if __name__ == '__main__':
    # Example Usage and Tests
    valid_expressions = [
        "1 + 2",
        "3 * (4 - 1)",
        "-5 / 2.0",
        "2**3",
        "10 // 3",
        "10 % 3",
        "+1 - -2",
        "3.14 * (2 + 0.5)",
        "(1 + 2) * 3 - 4 / 5 // 6 % 7**8" # Complex but valid
    ]

    invalid_expressions = [
        "import os",                           # Import statement
        "__import__('os').system('clear')",    # Harmful function call
        "print('hello')",                      # print function call
        "x + 1",                               # Variable 'x'
        "eval('1+1')",                         # eval function call
        "lambda x: x + 1",                     # Lambda function
        "a.b",                                 # Attribute access
        "l[0]",                                # Subscript access
        "1 + ",                                # Syntax error (caught by parse)
        "",                                    # Empty string
        "   ",                                 # Whitespace only string
        "1_000_000",                           # Valid number, but ast.Constant handles it
        "1+2; 3+4",                            # Multiple statements (SyntaxError in 'eval' mode)
        "def f(): return 1",                   # Function definition (SyntaxError in 'eval' mode)
    ]

    print("Testing valid expressions:")
    for expr in valid_expressions:
        is_valid = parse_user_expression(expr)
        print(f"'{expr}': {'Valid' if is_valid else 'Invalid'} (Expected: Valid)")
        assert is_valid, f"Expression '{expr}' should be valid."

    print("\nTesting invalid expressions:")
    for expr in invalid_expressions:
        is_valid = parse_user_expression(expr)
        print(f"'{expr}': {'Valid' if is_valid else 'Invalid'} (Expected: Invalid)")
        assert not is_valid, f"Expression '{expr}' should be invalid."

    # Test non-string input
    print("\nTesting non-string input:")
    non_string_input = 123
    is_valid = parse_user_expression(non_string_input)
    print(f"'{non_string_input}': {'Valid' if is_valid else 'Invalid'} (Expected: Invalid)")
    assert not is_valid, f"Input '{non_string_input}' should be invalid."

    print("\nAll parsing tests passed (if no assertions failed).")

    print("\nTesting evaluate_safe_expression:")
    test_eval_expressions = {
        "1 + 2": 3,
        "3 * (4 - 1)": 9,
        "-5 / 2.0": -2.5,
        "2**3": 8,
        "10 // 3": 3,
        "10 % 3": 1,
        "1 / 0": "Error: Division by zero.",
        "2**1000": "Error: Numerical result out of range.", # May depend on system float precision
        "import os": "Error: Invalid or unsafe expression.",
        "x + 1": "Error: Invalid or unsafe expression.",
        "1 +": "Error: Invalid or unsafe expression.", # Syntax error caught by parser
        "eval('1+1')": "Error: Invalid or unsafe expression."
    }

    for expr, expected in test_eval_expressions.items():
        result = evaluate_safe_expression(expr)
        print(f"Evaluating '{expr}': Result='{result}', Expected='{expected}'")
        if isinstance(expected, str) and expected.startswith("Error: Numerical result out of range."):
            # For OverflowError, the exact error message might vary slightly or not occur
            # if the system handles large numbers differently (e.g. Python's arbitrary precision integers)
            # So we check if the result is an error string for this specific case.
            # For 2**1000, Python integers handle this, so it won't be an OverflowError unless it's a float context.
            # Let's adjust the test case for 2**1000 to expect the large int.
            if expr == "2**1000": # Python handles large integers
                expected_val = 2**1000
                assert result == expected_val, f"Evaluation of '{expr}' failed. Got {result}, expected {expected_val}"
            elif isinstance(result, str) and "Error" in result:
                 pass # It's an error as expected
            else:
                assert result == expected, f"Evaluation of '{expr}' failed. Got {result}, expected {expected}"
        else:
            assert result == expected, f"Evaluation of '{expr}' failed. Got {result}, expected {expected}"
    
    # Test a very large number that should evaluate correctly with Python's arbitrary precision integers
    large_number_expr = "99999999999999999999999999999 * 99999999999999999999999999999"
    large_number_result = 99999999999999999999999999999 * 99999999999999999999999999999
    print(f"Evaluating '{large_number_expr}': Result='{evaluate_safe_expression(large_number_expr)}', Expected='{large_number_result}'")
    assert evaluate_safe_expression(large_number_expr) == large_number_result

    print("\nAll evaluation tests passed (if no assertions failed).")
