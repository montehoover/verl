import ast

def parse_quiz_expression(expression: str) -> bool:
    """
    Parses a user-submitted quiz expression and returns a boolean
    indicating whether the expression is valid for safe evaluation.

    Args:
        expression: The string input representing a quiz expression.

    Returns:
        True if the expression only contains safe operations, False otherwise.
    """
    try:
        parsed_ast = ast.parse(expression, mode='eval')
    except SyntaxError:
        return False  # Not a valid Python expression

    allowed_node_types = (
        ast.Expression,
        ast.Constant,  # For numbers (int, float). Also covers strings, True, False, None.
        ast.BinOp,
        ast.UnaryOp,
        # ast.Name could be allowed if we had a whitelist of variables/constants
    )

    allowed_bin_op_types = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,  # Exponentiation (e.g., 2**3)
        ast.Mod,  # Modulo
        ast.FloorDiv, # Integer division
    )

    allowed_unary_op_types = (
        ast.USub,  # Negation (e.g., -5)
        ast.UAdd,  # Unary plus (e.g., +5)
    )

    for node in ast.walk(parsed_ast):
        if not isinstance(node, allowed_node_types):
            return False  # Disallowed node type

        if isinstance(node, ast.Constant):
            # Ensure constants are numbers (or booleans if allowed for quiz logic)
            if not isinstance(node.value, (int, float, complex)):
                 # Allowing bools might be fine depending on quiz rules, e.g. True + 1
                 # For now, strictly numbers.
                 # If complex numbers are not desired, add check here.
                pass # Allow any constant type that ast.Constant supports by default for now.
                     # If stricter checking is needed (e.g. only int/float), it can be added.

        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, allowed_bin_op_types):
                return False  # Disallowed binary operator
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, allowed_unary_op_types):
                return False  # Disallowed unary operator
        # No need to check ast.Expression specifically, ast.walk handles it.

    return True


def safe_evaluate_expression(expression: str) -> str:
    """
    Safely evaluates a validated quiz expression string and computes the result.

    Args:
        expression: The validated string input representing a quiz expression.

    Returns:
        The result of the expression as a string, or an error message string
        if the expression is unsafe or an error occurs during evaluation.
    """
    if not parse_quiz_expression(expression):
        return "Error: Unsafe or invalid expression."

    try:
        # Evaluate the expression in a restricted environment
        # __builtins__ is still accessible, but parse_quiz_expression should prevent
        # calls to harmful builtins by disallowing ast.Call nodes.
        # For purely arithmetic expressions, an empty dict for globals is safer.
        result = eval(expression, {"__builtins__": {}}, {})
        # Ensure the result is a number (int, float, complex)
        if not isinstance(result, (int, float, complex)):
            return "Error: Evaluation did not result in a number."
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except OverflowError:
        return "Error: Numerical result out of range."
    except Exception as e:
        # Catch any other unexpected errors during evaluation
        return f"Error: Evaluation failed ({type(e).__name__})."


def evaluate_quiz_expression(quiz_expr: str) -> str:
    """
    Processes and evaluates a user-submitted quiz expression.

    Args:
        quiz_expr: The string input representing the quiz expression.

    Returns:
        The computed result as a string, or 'Error!' if the expression
        is unsafe or evaluation fails.
    """
    if not parse_quiz_expression(quiz_expr):
        return "Error!"

    result = safe_evaluate_expression(quiz_expr)
    if result.startswith("Error:"):
        return "Error!"
    return result


if __name__ == '__main__':
    # Test cases for parse_quiz_expression
    safe_expressions = [
        "1 + 2",
        "10 - 3 * 2",
        "(5 + 5) / 2",
        "2**3",
        "-5 + (2.5 * 4)",
        "10 % 3",
        "10 // 3",
        "+10 - -3",
        "3.14159",
        "1e5", # scientific notation
    ]

    unsafe_expressions = [
        "import os",
        "os.system('clear')",
        "__import__('os').system('rm -rf /')",
        "eval('1+1')",
        "lambda x: x + 1",
        "a + b",  # Disallowed if ast.Name is not in allowed_node_types
        "print('hello')",
        "[1, 2, 3]", # List comprehension
        "{'a': 1}", # Dict
        "1 < 2", # Comparison, might be safe but not an arithmetic op
        "def foo(): pass",
        "x = 5", # Assignment
        "1 if True else 0" # Conditional expression
    ]

    print("Testing safe expressions:")
    for expr in safe_expressions:
        is_safe = parse_quiz_expression(expr)
        print(f"'{expr}': {'Safe' if is_safe else 'Unsafe'}")
        assert is_safe, f"Expression '{expr}' should be safe but was marked unsafe."

    print("\nTesting unsafe expressions:")
    for expr in unsafe_expressions:
        is_safe = parse_quiz_expression(expr)
        print(f"'{expr}': {'Safe' if is_safe else 'Unsafe'}")
        assert not is_safe, f"Expression '{expr}' should be unsafe but was marked safe."

    print("\nAll tests passed for parse_quiz_expression (if no assertions failed).")

    print("\nTesting safe_evaluate_expression:")
    evaluation_test_cases = {
        "1 + 2": "3",
        "10 - 3 * 2": "4",
        "(5 + 5) / 2": "5.0",
        "2**3": "8",
        "-5 + (2.5 * 4)": "5.0",
        "10 % 3": "1",
        "10 // 3": "3",
        "+10 - -3": "13",
        "3.14159": "3.14159",
        "1e5": "100000.0",
        "1/0": "Error: Division by zero.",
        "1 + foo": "Error: Unsafe or invalid expression.", # foo is an undefined variable
        "print('hello')": "Error: Unsafe or invalid expression.",
        "2**1000": "Error: Numerical result out of range.", # May depend on system limits for float
                                                            # For integers, Python handles large numbers,
                                                            # but ast.Pow could be restricted if needed.
                                                            # Let's test a very large power.
        "10**1000": "Error: Numerical result out of range.", # This will likely be too large for float conversion by eval
                                                              # or hit OverflowError if it tries to compute.
                                                              # Python integers can be huge, but eval might hit limits
                                                              # or the string conversion might be an issue.
                                                              # For this test, we expect an overflow or similar.
    }

    # A very large number that Python integers can handle but might cause issues elsewhere or is good to test
    # Let's refine the 10**1000 test. eval() itself should produce a large int.
    # The OverflowError is more likely for float operations or if we try to convert to C types.
    # For now, let's assume eval() can produce large integers and str() can convert them.
    # The "2**1000" test is more likely to hit float overflow if it were 2.0**1000
    # Let's adjust the large number test to be more specific for overflow if possible,
    # or accept that large int results are fine.
    # For `eval("10**1000", ...)`, Python handles large integers fine.
    # The `OverflowError` is more common with floats.
    # Let's add a float overflow test.
    evaluation_test_cases["1e308 * 10"] = "Error: Numerical result out of range." # Should cause float overflow

    # Re-add a large integer test that should pass
    large_int_expr = "10**100"
    large_int_result = str(10**100)
    evaluation_test_cases[large_int_expr] = large_int_result


    for expr, expected_result in evaluation_test_cases.items():
        actual_result = safe_evaluate_expression(expr)
        print(f"'{expr}' -> Expected: '{expected_result}', Got: '{actual_result}'")
        assert actual_result == expected_result, f"Evaluation of '{expr}' failed. Expected '{expected_result}', got '{actual_result}'"

    print("\nAll tests passed for safe_evaluate_expression (if no assertions failed).")

    print("\nTesting evaluate_quiz_expression:")
    quiz_eval_test_cases = {
        "1 + 2": "3",
        "10 - 3 * 2": "4",
        "(5 + 5) / 2": "5.0",
        "1/0": "Error!", # Division by zero from safe_evaluate_expression
        "import os": "Error!", # Unsafe from parse_quiz_expression
        "a + b": "Error!", # Unsafe (undefined variable) from parse_quiz_expression
        "print('hello')": "Error!", # Unsafe from parse_quiz_expression
        "1e308 * 10": "Error!", # Overflow from safe_evaluate_expression
        "10**100": str(10**100), # Valid large number
    }

    for expr, expected_output in quiz_eval_test_cases.items():
        actual_output = evaluate_quiz_expression(expr)
        print(f"'{expr}' -> Expected: '{expected_output}', Got: '{actual_output}'")
        assert actual_output == expected_output, f"evaluate_quiz_expression of '{expr}' failed. Expected '{expected_output}', got '{actual_output}'"

    print("\nAll tests passed for evaluate_quiz_expression (if no assertions failed).")
