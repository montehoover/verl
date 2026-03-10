import ast
import operator

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def evaluate_simple_expression(expression_string: str, variables: dict = None) -> str:
    """
    Evaluates a string representing a basic mathematical expression,
    optionally with variables, and returns the calculated result as a string.

    The expression can include numbers, variables, and basic arithmetic
    operations: +, -, *, /.

    Args:
        expression_string: The mathematical expression to evaluate.
        variables: A dictionary mapping variable names (str) to their
                   numeric values (int or float). Defaults to None or an empty dict.

    Returns:
        The result of the calculation as a string.

    Raises:
        ValueError: If the expression is invalid, contains unsupported operations,
                    or cannot be processed.
    """
    try:
        # Parse the expression string into an AST (Abstract Syntax Tree)
        node = ast.parse(expression_string, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {expression_string}") from e

    _variables = variables or {}

    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.Num): # Deprecated in Python 3.8+
            return node.n
        elif isinstance(node, ast.Constant): # For Python 3.8+
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"Unsupported constant type in expression: {type(node.value)}")
            return node.value
        elif isinstance(node, ast.Name): # Handle variables
            var_name = node.id
            if var_name in _variables:
                val = _variables[var_name]
                if not isinstance(val, (int, float)):
                    raise ValueError(f"Variable '{var_name}' has non-numeric value: {val}")
                return val
            else:
                raise ValueError(f"Undefined variable: '{var_name}'")
        elif isinstance(node, ast.BinOp):
            left_val = _eval_node(node.left)
            right_val = _eval_node(node.right)
            op_type = type(node.op)
            if op_type in _ALLOWED_OPERATORS:
                operator_func = _ALLOWED_OPERATORS[op_type]
                if op_type == ast.Div and right_val == 0:
                    raise ValueError("Division by zero is not allowed.")
                return operator_func(left_val, right_val)
            else:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand_val = _eval_node(node.operand)
            return -operand_val
        else:
            raise ValueError(f"Unsupported expression structure: {ast.dump(node)}")

    try:
        result = _eval_node(node)
        if not isinstance(result, (int, float)): # Should be caught earlier, but good failsafe
            raise ValueError("Expression did not evaluate to a number.")
        return str(result)
    except ValueError: # Re-raise ValueErrors from _eval_node
        raise
    except ZeroDivisionError: # Should be caught by the check in _eval_node, but as a fallback
        raise ValueError("Division by zero is not allowed.")
    except Exception as e:
        raise ValueError(f"Could not evaluate expression: {expression_string}. Error: {e}") from e

if __name__ == '__main__':
    # Example Usage and Basic Tests
    # Each key is a tuple: (expression_string, variables_dict_or_None)
    # Value is the expected string result or "ValueError"
    test_cases = {
        ("1 + 1", None): "2.0",
        ("5 - 2", None): "3.0",
        ("3 * 4", None): "12.0",
        ("10 / 2", None): "5.0",
        ("2 + 3 * 4", None): "14.0",
        ("(2 + 3) * 4", None): "20.0",
        ("10 / 0", None): "ValueError",
        ("1 +", None): "ValueError",
        ("1 + foo", None): "ValueError", # foo not in vars
        ("1 + (2 * 3", None): "ValueError",
        ("-5 + 2", None): "-3.0",
        ("-(5 + 2)", None): "-7.0",
        ("10.5 * 2", None): "21.0",
        ("import os", None): "ValueError",
        ("__import__('os').system('echo unsafe')", None): "ValueError",

        # Tests with variables
        ("x + y", {"x": 5, "y": 3}): "8.0",
        ("x * x", {"x": 2.5}): "6.25",
        ("radius * 2 * pi", {"radius": 10, "pi": 3.14159}): "62.8318",
        ("a + b / c", {"a": 1, "b": 10, "c": 2}): "6.0",
        ("(a + b) / c", {"a": 1, "b": 9, "c": 2}): "5.0",
        ("x / 0", {"x": 5}): "ValueError", # Division by zero with var
        ("x + y", {"x": 5}): "ValueError", # y undefined
        ("x + y", {"x": 5, "y": "text"}): "ValueError", # y not a number
        ("z", {}): "ValueError", # z undefined, empty vars
        ("my_var", {"my_var": -100}): "-100.0",
    }

    for (expr, variables), expected in test_cases.items():
        try:
            result = evaluate_simple_expression(expr, variables)
            if result == expected:
                print(f"PASS: '{expr}' with vars {variables} -> '{result}'")
            else:
                print(f"FAIL: '{expr}' with vars {variables} -> '{result}', expected '{expected}'")
        except ValueError as e:
            if expected == "ValueError":
                print(f"PASS (ValueError expected): '{expr}' with vars {variables} -> {e}")
            else:
                print(f"FAIL (Unexpected ValueError): '{expr}' with vars {variables} -> {e}, expected '{expected}'")
        except Exception as e:
            print(f"FAIL (Unexpected Exception): '{expr}' with vars {variables} -> {e}")

    print("\nTesting with Python's eval for comparison (where safe and applicable):")
    # (expression_string, variables_dict, expected_eval_string_for_comparison)
    safe_eval_test_cases = [
        ("1 + 1", None, "1 + 1"),
        ("x + y", {"x": 5, "y": 3}, "5 + 3"),
        ("radius * 2 * pi", {"radius": 10, "pi": 3.14}, "10 * 2 * 3.14"), # Simpler pi for direct eval
        ("(a + b) / c", {"a": 1, "b": 9, "c": 2}, "(1 + 9) / 2"),
    ]

    for expr_str, var_dict, eval_str in safe_eval_test_cases:
        try:
            custom_eval_res_str = evaluate_simple_expression(expr_str, var_dict)
            custom_eval_res = float(custom_eval_res_str)
            # For eval, we need to provide the context if variables are used
            # This is a simplified way, not robust for all var names in eval's locals
            python_eval_res = float(eval(eval_str, {}, var_dict or {})) # Use with caution

            if abs(custom_eval_res - python_eval_res) < 1e-9:
                print(f"COMPARE PASS: '{expr_str}' (vars: {var_dict}) -> Custom: {custom_eval_res}, Python eval ('{eval_str}'): {python_eval_res}")
            else:
                print(f"COMPARE FAIL: '{expr_str}' (vars: {var_dict}) -> Custom: {custom_eval_res}, Python eval ('{eval_str}'): {python_eval_res}")
        except Exception as e:
            print(f"COMPARE ERROR for '{expr_str}' (vars: {var_dict}): {e}")
