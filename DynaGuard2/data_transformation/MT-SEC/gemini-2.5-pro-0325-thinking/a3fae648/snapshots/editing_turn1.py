import ast
import operator

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def evaluate_expression(expression: str) -> str:
    """
    Evaluates a string representing a simple mathematical expression
    and returns the calculated result as a string.

    Supports basic arithmetic operations: +, -, *, /.
    Raises ValueError if the expression is invalid or cannot be processed.
    """
    try:
        # Parse the expression to an Abstract Syntax Tree (AST)
        node = ast.parse(expression, mode='eval')

        def _eval_node(node):
            if isinstance(node, ast.Num): # Python 3.7 and earlier
                return node.n
            if isinstance(node, ast.Constant): # Python 3.8 and later
                return node.value
            elif isinstance(node, ast.BinOp):
                left_val = _eval_node(node.left)
                right_val = _eval_node(node.right)
                op_type = type(node.op)
                if op_type in _ALLOWED_OPS:
                    if op_type == ast.Div and right_val == 0:
                        raise ZeroDivisionError("division by zero")
                    return _ALLOWED_OPS[op_type](left_val, right_val)
                else:
                    raise ValueError(f"Unsupported operator: {op_type.__name__}")
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                operand_val = _eval_node(node.operand)
                return -operand_val
            elif isinstance(node, ast.Expression): # The top level node
                return _eval_node(node.body)
            else:
                raise ValueError(f"Unsupported expression structure: {type(node).__name__}")

        result = _eval_node(node)
        
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
        "__import__('os').system('echo unsafe')": "ValueError" # Security check
    }

    for expr, expected in test_expressions.items():
        try:
            result = evaluate_expression(expr)
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

    print("\nTesting edge cases:")
    try:
        evaluate_expression("1 / 0")
    except ValueError as e:
        print(f"PASS (ZeroDivisionError): '1 / 0' -> {e}")

    try:
        evaluate_expression("1 + '2'")
    except ValueError as e:
        print(f"PASS (TypeError caught): \"1 + '2'\" -> {e}")
    
    try:
        evaluate_expression("abs(-1)") # Example of a disallowed function
    except ValueError as e:
        print(f"PASS (Disallowed function): 'abs(-1)' -> {e}")

    print(evaluate_expression("2.5 * 2"))
    print(evaluate_expression("100"))
    print(evaluate_expression("-3"))
