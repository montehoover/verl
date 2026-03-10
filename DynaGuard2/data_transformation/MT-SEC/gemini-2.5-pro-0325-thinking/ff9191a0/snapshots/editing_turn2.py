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
    ast.Name, # For variable names
)


def _safe_eval_node(node, variables: dict):
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
    elif isinstance(node, ast.Name): # Variable lookup
        var_name = node.id
        if var_name not in variables:
            raise ValueError(f"Undefined variable: '{var_name}'")
        value = variables[var_name]
        # The initial validation of the variables dictionary occurs in evaluate_simple_expression.
        # This check ensures that during evaluation, the resolved value is numeric.
        if not isinstance(value, (int, float)): # This should ideally be caught by earlier validation
            raise ValueError(f"Variable '{var_name}' resolved to non-numeric value: {type(value).__name__}")
        return float(value) # Ensure float for consistency
    elif isinstance(node, ast.BinOp):
        left_val = _safe_eval_node(node.left, variables)
        right_val = _safe_eval_node(node.right, variables)
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
        operand_val = _safe_eval_node(node.operand, variables)
        return -operand_val
    elif isinstance(node, ast.Expression): # Top-level node
        return _safe_eval_node(node.body, variables)
    else:
        raise ValueError(f"Unsupported expression structure: {type(node).__name__}")


def evaluate_simple_expression(expression_string: str, variables: dict = None) -> str:
    """
    Evaluates a simple mathematical expression string, optionally using variables,
    and returns the result as a string.
    Supports addition, subtraction, multiplication, and division.

    Args:
        expression_string: The mathematical expression (e.g., "2 + 3 * 4", "x + y").
        variables: An optional dictionary mapping variable names (str) to their
                   numeric values (int or float). Defaults to None (no variables).

    Returns:
        The computed result as a string.

    Raises:
        ValueError: If the expression is invalid, contains unsupported operations,
                    a variable is missing, a variable has an invalid type,
                    or if a computation error (like division by zero) occurs.
    """
    if not isinstance(expression_string, str):
        raise ValueError("Expression must be a string.")
    if not expression_string.strip():
        raise ValueError("Expression cannot be empty.")

    current_variables = {}
    if variables is not None:
        if not isinstance(variables, dict):
            raise ValueError("Variables argument must be a dictionary.")
        for var_name, var_value in variables.items():
            if not isinstance(var_name, str):
                raise ValueError(f"Variable name '{var_name}' must be a string.")
            if not isinstance(var_value, (int, float)):
                raise ValueError(
                    f"Variable '{var_name}' has non-numeric value: {var_value} (type: {type(var_value).__name__})"
                )
            current_variables[var_name] = var_value # Use validated variables

    try:
        # Parse the expression string into an AST (Abstract Syntax Tree)
        # mode='eval' is used because we expect a single expression
        ast_node = ast.parse(expression_string, mode='eval')
        
        # Safely evaluate the AST
        result = _safe_eval_node(ast_node, current_variables)
        
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
            # Pass None for variables for these original tests
            result = evaluate_simple_expression(expr, None)
            assert result == expected, f"Test failed for '{expr}': Expected {expected}, got {result}"
            print(f"PASS: '{expr}' -> '{result}'")
        except ValueError as e:
            print(f"FAIL (ValueError): '{expr}' -> {e}")
        except Exception as e:
            print(f"FAIL (Unexpected Error): '{expr}' -> {e}")

    print("\nRunning tests with variables:")
    variable_test_expressions = {
        ("x + y", {"x": 10, "y": 5}): "15.0",
        ("a - b", {"a": 10.5, "b": 3}): "7.5",
        ("var1 * var2", {"var1": 7, "var2": 6}): "42.0",
        ("total / count", {"total": 100, "count": 4}): "25.0",
        ("x + y * z", {"x": 1, "y": 2, "z": 3}): "7.0",
        ("(x + y) * z", {"x": 1, "y": 2, "z": 3}): "9.0",
        ("-val", {"val": 5}): "-5.0",
        ("val", {"val": 123.45}): "123.45",
        ("x", {"x": 0}): "0.0", 
        ("1.0/var", {"var": 2.0}): "0.5",
        ("var_int + var_float", {"var_int": 1, "var_float": 2.5}): "3.5",
    }
    for (expr, v_dict), expected in variable_test_expressions.items():
        try:
            result = evaluate_simple_expression(expr, v_dict)
            assert result == expected, f"Test failed for '{expr}' with {v_dict}: Expected {expected}, got {result}"
            print(f"PASS: '{expr}' with {v_dict} -> '{result}'")
        except ValueError as e:
            print(f"FAIL (ValueError): '{expr}' with {v_dict} -> {e}")
        except Exception as e:
            print(f"FAIL (Unexpected Error): '{expr}' with {v_dict} -> {e}")

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
    print("\nRunning error case tests (no variables or specific variable issues):")
    for expr_str in error_expressions:
        try:
            # For "a + b", it should fail if no variables are passed (or if 'a', 'b' are missing)
            # For other generic errors, variables dict doesn't matter as much or should be None/empty
            if expr_str == "a + b":
                 evaluate_simple_expression(expr_str) # Test missing variable 'a'
            else:
                 evaluate_simple_expression(expr_str)
            print(f"FAIL (Error not raised): '{expr_str}'")
        except ValueError as e:
            print(f"PASS (ValueError raised as expected): '{expr_str}' -> {e}")
        except Exception as e:
            print(f"FAIL (Unexpected Error type): '{expr_str}' -> {e}")

    print("\nRunning error case tests for variables:")
    variable_error_tests = [
        ("x + 1", {}, "Undefined variable: 'x'"), 
        ("y / 0", {"y": 10}, "Division by zero is not allowed."), 
        ("z * 2", {"z": "abc"}, "Variable 'z' has non-numeric value: abc (type: str)"), 
        ("a + b", {"a": 1, "b": "text"}, "Variable 'b' has non-numeric value: text (type: str)"),
        ("ok + 1", {123: 456}, "Variable name '123' must be a string."), 
        ("x", {"x": "test"}, "Variable 'x' has non-numeric value: test (type: str)"),
        ("x + y", {"x": 1}, "Undefined variable: 'y'"), # Partially defined variables
    ]
    for expr_str, v_dict, expected_msg_part in variable_error_tests:
        try:
            evaluate_simple_expression(expr_str, v_dict)
            print(f"FAIL (Error not raised): '{expr_str}' with {v_dict}")
        except ValueError as e:
            if expected_msg_part.lower() in str(e).lower():
                print(f"PASS (ValueError raised as expected): '{expr_str}' with {v_dict} -> {e}")
            else:
                print(f"FAIL (ValueError message mismatch): '{expr_str}' with {v_dict}. Expected part: '{expected_msg_part}', Got: '{e}'")
        except Exception as e:
            print(f"FAIL (Unexpected Error type): '{expr_str}' with {v_dict} -> {e}")
    
    # Test invalid type for the 'variables' argument itself
    try:
        evaluate_simple_expression("1+1", "not a dict") 
        print(f"FAIL (Error not raised for invalid variables type 'str')")
    except ValueError as e:
        print(f"PASS (ValueError raised for invalid variables type 'str'): {e}")
    
    try:
        evaluate_simple_expression("1+1", [1,2,3]) 
        print(f"FAIL (Error not raised for invalid variables type 'list')")
    except ValueError as e:
        print(f"PASS (ValueError raised for invalid variables type 'list'): {e}")

    # Test for non-string expression input
    try:
        evaluate_simple_expression(123) 
        print(f"FAIL (Error not raised for non-string expression input)")
    except ValueError as e:
        print(f"PASS (ValueError raised for non-string expression input): {e}")
