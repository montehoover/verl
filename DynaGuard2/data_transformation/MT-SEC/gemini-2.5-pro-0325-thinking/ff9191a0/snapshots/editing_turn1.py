import ast
import operator

# A dictionary to map AST nodes to actual operations
_SUPPORTED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

# A set of supported AST node types for safety
_SUPPORTED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp, # For negative numbers
    ast.Num, # Deprecated in Python 3.8, use ast.Constant
    ast.Constant, # For numbers and potentially other constants
    ast.NameConstant, # For True, False, None (not used here but good for completeness)
    ast.Load, # Context for loading a variable (not directly used but part of AST structure)
    ast.USub, # For unary minus
)


def _safe_eval_node(node):
    """
    Recursively evaluates an AST node, supporting only basic arithmetic.
    """
    if not isinstance(node, tuple(_SUPPORTED_NODE_TYPES)):
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        return node.value
    elif isinstance(node, ast.Num): # For Python < 3.8
        if not isinstance(node.n, (int, float)):
            raise ValueError(f"Unsupported number type: {type(node.n).__name__}")
        return node.n
    elif isinstance(node, ast.BinOp):
        left_val = _safe_eval_node(node.left)
        right_val = _safe_eval_node(node.right)
        op_func = _SUPPORTED_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        try:
            return op_func(left_val, right_val)
        except ZeroDivisionError:
            raise ValueError("Division by zero is not allowed.")
        except Exception as e:
            raise ValueError(f"Error during operation: {e}")
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand_val = _safe_eval_node(node.operand)
        return -operand_val
    elif isinstance(node, ast.Expression): # Top-level node
        return _safe_eval_node(node.body)
    else:
        raise ValueError(f"Unsupported expression structure: {type(node).__name__}")


def evaluate_simple_expression(expression_string: str) -> str:
    """
    Evaluates a simple mathematical expression string and returns the result as a string.
    Supports addition, subtraction, multiplication, and division.
    Raises ValueError for invalid expressions or computation failures.

    Args:
        expression_string: The mathematical expression (e.g., "2 + 3 * 4").

    Returns:
        The computed result as a string.

    Raises:
        ValueError: If the expression is invalid, contains unsupported operations,
                    or if a computation error (like division by zero) occurs.
    """
    if not isinstance(expression_string, str):
        raise ValueError("Expression must be a string.")
    if not expression_string.strip():
        raise ValueError("Expression cannot be empty.")

    try:
        # Parse the expression string into an AST (Abstract Syntax Tree)
        # mode='eval' is used because we expect a single expression
        ast_node = ast.parse(expression_string, mode='eval')
        
        # Safely evaluate the AST
        result = _safe_eval_node(ast_node)
        
        # Ensure the result is a number before converting to string
        if not isinstance(result, (int, float)):
            raise ValueError("Expression did not evaluate to a number.")
            
        return str(result)
    except SyntaxError:
        raise ValueError("Invalid syntax in expression.")
    except ValueError: # Re-raise ValueErrors from _safe_eval_node
        raise
    except Exception as e:
        # Catch any other unexpected errors during parsing or evaluation
        raise ValueError(f"Failed to evaluate expression: {e}")

if __name__ == '__main__':
    # Example Usage and Basic Tests
    test_expressions = {
        "2 + 3": "5.0",
        "10 - 4": "6.0",
        "5 * 6": "30.0",
        "20 / 4": "5.0",
        "2 + 3 * 4": "14.0",
        "(2 + 3) * 4": "20.0",
        "10 / 2 - 1": "4.0",
        "-5 + 10": "5.0",
        "3 * -2": "-6.0",
        "1 / 3": str(1/3),
        "  1 +   1  ": "2.0"
    }

    print("Running tests for evaluate_simple_expression:")
    for expr, expected in test_expressions.items():
        try:
            result = evaluate_simple_expression(expr)
            assert result == expected, f"Test failed for '{expr}': Expected {expected}, got {result}"
            print(f"PASS: '{expr}' -> '{result}'")
        except ValueError as e:
            print(f"FAIL (ValueError): '{expr}' -> {e}")
        except Exception as e:
            print(f"FAIL (Unexpected Error): '{expr}' -> {e}")

    error_expressions = [
        "2 + ",
        "10 / 0",
        "a + b",
        "2 ** 3", # Exponentiation not supported by default
        "import os",
        "eval('1+1')",
        "my_func()",
        "",
        "   ",
        "1 + (2 * 3", # Mismatched parentheses
        "1 + 2)",
        "1 + / 2",
    ]
    print("\nRunning error case tests:")
    for expr in error_expressions:
        try:
            evaluate_simple_expression(expr)
            print(f"FAIL (Error not raised): '{expr}'")
        except ValueError as e:
            print(f"PASS (ValueError raised as expected): '{expr}' -> {e}")
        except Exception as e:
            print(f"FAIL (Unexpected Error type): '{expr}' -> {e}")
    
    try:
        evaluate_simple_expression(123) # Non-string input
        print(f"FAIL (Error not raised for non-string input)")
    except ValueError as e:
        print(f"PASS (ValueError raised for non-string input): {e}")
