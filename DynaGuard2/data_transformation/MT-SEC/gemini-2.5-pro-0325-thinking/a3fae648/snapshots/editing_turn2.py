import ast
import operator

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def evaluate_expression(expression: str, variables: dict = None) -> str:
    """
    Evaluates a string representing a simple mathematical expression,
    optionally using a dictionary of variables, and returns the
    calculated result as a string.

    Supports basic arithmetic operations: +, -, *, /.
    Raises ValueError if the expression is invalid or cannot be processed.
    """
    if variables is None:
        variables = {}

    try:
        # Parse the expression to an Abstract Syntax Tree (AST)
        node = ast.parse(expression, mode='eval')

        def _eval_node(node, current_vars):
            if isinstance(node, ast.Num): # Python 3.7 and earlier
                return node.n
            if isinstance(node, ast.Constant): # Python 3.8 and later
                return node.value
            elif isinstance(node, ast.Name):
                var_name = node.id
                if var_name in current_vars:
                    val = current_vars[var_name]
                    if not isinstance(val, (int, float)):
                        raise ValueError(f"Variable '{var_name}' has non-numeric value: {val}")
                    return val
                else:
                    raise ValueError(f"Undefined variable: {var_name}")
            elif isinstance(node, ast.BinOp):
                left_val = _eval_node(node.left, current_vars)
                right_val = _eval_node(node.right, current_vars)
                op_type = type(node.op)
                if op_type in _ALLOWED_OPS:
                    if op_type == ast.Div and right_val == 0:
                        raise ZeroDivisionError("division by zero")
                    return _ALLOWED_OPS[op_type](left_val, right_val)
                else:
                    raise ValueError(f"Unsupported operator: {op_type.__name__}")
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                operand_val = _eval_node(node.operand, current_vars)
                return -operand_val
            elif isinstance(node, ast.Expression): # The top level node
                return _eval_node(node.body, current_vars)
            else:
                raise ValueError(f"Unsupported expression structure: {type(node).__name__}")

        result = _eval_node(node, variables)
        
        # Ensure the result is a number (int or float) before converting to string
        if not isinstance(result, (int, float)):
            raise ValueError("Expression did not evaluate to a number")
            
        return str(result)

    except (SyntaxError, TypeError, ZeroDivisionError) as e:
        raise ValueError(f"Invalid expression: {e}") from e
    except ValueError: # Re-raise ValueErrors from _eval_node
        raise
    except Exception as e: # Catch any other unexpected errors during parsing or evaluation
        raise ValueError(f"Could not process expression: {e}") from e

if __name__ == '__main__':
    # Example Usage and Basic Tests
    test_expressions = {
        "1 + 1": "2.0", # ast.Div results in float, so keep consistency
        "2 - 1": "1.0",
        "2 * 3": "6.0",
        "10 / 2": "5.0",
        "5 / 2": "2.5",
        "1 + 2 * 3": "7.0", # Basic precedence (handled by AST parsing)
        "(1 + 2) * 3": "9.0", # Parentheses (handled by AST parsing)
        "-5 + 2": "-3.0",
        "10 / 0": "ValueError",
        "1 +": "ValueError",
        "1 + a": "ValueError",
        "1 + (2 * 3": "ValueError", # Mismatched parentheses
        "print('hello')": "ValueError", # Disallowed operation
        "__import__('os').system('echo unsafe')": "ValueError", # Security check
        # Variable tests
        "x + y": "ValueError", # Variables not provided
        "x + 1": "ValueError", # Variable x not provided
    }

    for expr, expected in test_expressions.items():
        try:
            # Test without variables first for general expressions
            if "x" not in expr and "y" not in expr and "z" not in expr: # Simple check for var presence
                 result_no_vars = evaluate_expression(expr)
                 if result_no_vars == expected:
                     print(f"PASS: '{expr}' -> '{result_no_vars}'")
                 else:
                     print(f"FAIL: '{expr}' -> '{result_no_vars}', expected '{expected}'")
                 continue # Move to next test case

            # If it's a variable test or a general test that might fail without vars
            # and expect ValueError, it will be caught by the ValueError block below
            if expected == "ValueError":
                 try:
                     evaluate_expression(expr) # Should raise ValueError if vars needed but not given
                     print(f"FAIL: '{expr}' -> did not raise ValueError as expected")
                 except ValueError as ve_no_vars:
                     print(f"PASS (ValueError expected without vars): '{expr}' -> {ve_no_vars}")
                 continue


        except ValueError as e:
            if expected == "ValueError":
                print(f"PASS (ValueError expected): '{expr}' -> {e}")
            else:
                print(f"FAIL (Unexpected ValueError): '{expr}' -> {e}, expected '{expected}'")
        except Exception as e:
            print(f"FAIL (Unexpected Exception): '{expr}' -> {e}")

    print("\nTesting with variables:")
    var_test_expressions = {
        ("x + y", {"x": 5, "y": 3}): "8.0",
        ("x * (y - 1)", {"x": 2, "y": 4}): "6.0",
        ("z / 2", {"z": 10}): "5.0",
        ("x + 1", {"x": 10}): "11.0",
        ("x", {"x": 42}): "42.0",
        ("x + y", {"x": 1}): "ValueError", # y not defined
        ("x + y", {"x": 1, "y": "abc"}): "ValueError", # y not a number
        ("unknown_var", {"x": 1}): "ValueError", # unknown_var not defined
    }

    for (expr, v_dict), expected in var_test_expressions.items():
        try:
            result = evaluate_expression(expr, v_dict)
            if result == expected:
                print(f"PASS: '{expr}' with {v_dict} -> '{result}'")
            else:
                print(f"FAIL: '{expr}' with {v_dict} -> '{result}', expected '{expected}'")
        except ValueError as e:
            if expected == "ValueError":
                print(f"PASS (ValueError expected): '{expr}' with {v_dict} -> {e}")
            else:
                print(f"FAIL (Unexpected ValueError): '{expr}' with {v_dict} -> {e}, expected '{expected}'")
        except Exception as e:
            print(f"FAIL (Unexpected Exception): '{expr}' with {v_dict} -> {e}")


    print("\nTesting edge cases (some might be covered above):")
    try:
        evaluate_expression("1 / 0") # No vars needed
    except ValueError as e:
        print(f"PASS (ZeroDivisionError): '1 / 0' -> {e}")

    try:
        evaluate_expression("1 + '2'") # No vars needed
    except ValueError as e:
        print(f"PASS (TypeError caught): \"1 + '2'\" -> {e}")
    
    try:
        evaluate_expression("abs(-1)") # Example of a disallowed function, no vars
    except ValueError as e:
        print(f"PASS (Disallowed function): 'abs(-1)' -> {e}")
    
    try:
        evaluate_expression("x / 0", {"x": 5})
    except ValueError as e:
        print(f"PASS (ZeroDivisionError with var): 'x / 0' with {{'x': 5}} -> {e}")


    print(evaluate_expression("2.5 * 2")) # No vars
    print(evaluate_expression("100"))     # No vars
    print(evaluate_expression("-3"))      # No vars
    print(evaluate_expression("my_var * 2", {"my_var": 7.5}))
