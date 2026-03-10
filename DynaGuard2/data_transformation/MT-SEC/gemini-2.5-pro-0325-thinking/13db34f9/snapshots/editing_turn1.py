import ast
import operator

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def evaluate_simple_expression(expression_string: str) -> str:
    """
    Evaluates a string representing a basic mathematical expression
    and returns the calculated result as a string.

    The expression can include numbers and basic arithmetic operations: +, -, *, /.

    Args:
        expression_string: The mathematical expression to evaluate.

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

    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.Num): # In Python 3.8+, ast.Constant for numbers
            return node.n
        elif isinstance(node, ast.Constant): # For Python 3.8+
             if not isinstance(node.value, (int, float)):
                 raise ValueError(f"Unsupported constant type in expression: {type(node.value)}")
             return node.value
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
        if not isinstance(result, (int, float)):
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
    test_expressions = {
        "1 + 1": "2.0", # ast.Div results in float, so we make others consistent
        "5 - 2": "3.0",
        "3 * 4": "12.0",
        "10 / 2": "5.0",
        "2 + 3 * 4": "14.0", # Standard operator precedence
        "(2 + 3) * 4": "20.0", # Parentheses
        "10 / 0": "ValueError",
        "1 +": "ValueError",
        "1 + foo": "ValueError",
        "1 + (2 * 3": "ValueError", # Mismatched parentheses
        "-5 + 2": "-3.0",
        "-(5 + 2)": "-7.0",
        "10.5 * 2": "21.0",
        "import os": "ValueError", # Test against disallowed operations
        "__import__('os').system('echo unsafe')": "ValueError",
    }

    for expr, expected in test_expressions.items():
        try:
            result = evaluate_simple_expression(expr)
            if result == expected:
                print(f"PASS: '{expr}' -> '{result}'")
            else:
                print(f"FAIL: '{expr}' -> '{result}', expected '{expected}'")
        except ValueError as e:
            if expected == "ValueError":
                print(f"PASS (ValueError expected): '{expr}' -> {e}")
            else:
                print(f"FAIL (Unexpected ValueError): '{expr}' -> {e}, expected '{expected}'")
        except Exception as e:
            print(f"FAIL (Unexpected Exception): '{expr}' -> {e}")

    print("\nTesting with Python's eval for comparison (where safe):")
    safe_expressions_for_eval = [
        "1 + 1", "5 - 2", "3 * 4", "10 / 2", "2 + 3 * 4", "(2 + 3) * 4", "-5 + 2", "-(5 + 2)", "10.5 * 2"
    ]
    for expr_str in safe_expressions_for_eval:
        try:
            custom_eval_res_str = evaluate_simple_expression(expr_str)
            # Convert to float for comparison to avoid "2" vs "2.0" issues
            custom_eval_res = float(custom_eval_res_str)
            python_eval_res = float(eval(expr_str)) # Use with caution
            if abs(custom_eval_res - python_eval_res) < 1e-9 : # Compare floats
                print(f"COMPARE PASS: '{expr_str}' -> Custom: {custom_eval_res}, Python eval: {python_eval_res}")
            else:
                print(f"COMPARE FAIL: '{expr_str}' -> Custom: {custom_eval_res}, Python eval: {python_eval_res}")
        except Exception as e:
            print(f"COMPARE ERROR for '{expr_str}': {e}")
