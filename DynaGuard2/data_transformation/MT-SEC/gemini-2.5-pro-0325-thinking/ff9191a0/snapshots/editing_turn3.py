import re
import operator
import ast

# --- AST-based Safe Evaluator ---

# Allowed mathematical operators
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    # ast.Pow: operator.pow, # Example: if you want to support exponentiation
    # ast.USub: operator.neg, # Unary minus is handled separately for now
}

# Allowed AST node types for safety.
# ast.Load is a context, ast.NameConstant for True/False/None (not used here)
# ast.Num is for Python < 3.8, ast.Constant for Python 3.8+
ALLOWED_NODE_TYPES = (
    ast.Expression,  # The top-level node for an expression.
    ast.Constant,    # For numeric literals (Python 3.8+).
    ast.Num,         # For numeric literals (Python < 3.8).
    ast.Name,        # For variable names.
    ast.BinOp,       # For binary operations (e.g., +, -, *, /).
    ast.UnaryOp,     # For unary operations (e.g., -).
    ast.USub,        # Specific unary operator for negation.
    ast.Load,        # Indicates that a variable is being loaded (read).
)


def _recursive_ast_eval(node, variable_values: dict):
    """
    Recursively evaluates an AST node, supporting basic arithmetic and variables.
    Raises ValueError for unsupported nodes, operations, or errors.
    """
    if not isinstance(node, ALLOWED_NODE_TYPES):
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

    if isinstance(node, (ast.Constant, ast.Num)): # Numeric literal
        value = node.value if isinstance(node, ast.Constant) else node.n
        if not isinstance(value, (int, float)):
            raise ValueError(f"Unsupported constant type: {type(value).__name__}")
        return float(value) # Standardize to float for calculations
    elif isinstance(node, ast.Name): # Variable
        var_name = node.id
        if var_name not in variable_values:
            raise ValueError(f"Undefined variable: '{var_name}'")
        # Value type is already validated in evaluate_expression_safely
        return float(variable_values[var_name]) # Standardize to float
    elif isinstance(node, ast.BinOp): # Binary operation
        left_val = _recursive_ast_eval(node.left, variable_values)
        right_val = _recursive_ast_eval(node.right, variable_values)
        op_func = ALLOWED_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        try:
            return op_func(left_val, right_val)
        except ZeroDivisionError:
            raise ValueError("Division by zero is not allowed.")
        except Exception as e: # Catch other potential math errors
            raise ValueError(f"Error during binary operation: {e}")
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): # Unary minus
        operand_val = _recursive_ast_eval(node.operand, variable_values)
        return -operand_val
    elif isinstance(node, ast.Expression): # Top-level expression node
        return _recursive_ast_eval(node.body, variable_values)
    else:
        # This case should ideally be caught by the initial isinstance check,
        # but serves as a fallback.
        raise ValueError(f"Unsupported or malformed AST node: {type(node).__name__}")


def evaluate_expression_safely(math_expression: str, variable_mapping: dict) -> str:
    """
    Securely evaluates a mathematical expression string with variables.

    Args:
        math_expression: The mathematical expression string (e.g., "2 * x + y").
        variable_mapping: A dictionary mapping variable names (str) to their
                          numeric values (int or float). Can be empty if no
                          variables are in the expression.

    Returns:
        The computed result as a string.

    Raises:
        ValueError: If the input expression is invalid (e.g., syntax error,
                    unsupported operations, undefined variables, non-numeric
                    variable values) or if any computation error occurs (e.g.,
                    division by zero).
    """
    if not isinstance(math_expression, str):
        raise ValueError("Math expression must be a string.")
    if not math_expression.strip():
        raise ValueError("Math expression cannot be empty.")

    if not isinstance(variable_mapping, dict):
        raise ValueError("Variable mapping must be a dictionary.")

    # Validate and prepare variable_mapping
    current_variable_values = {}
    for var_name, var_value in variable_mapping.items():
        if not isinstance(var_name, str):
            raise ValueError(f"Variable name '{var_name}' must be a string.")
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", var_name):
             raise ValueError(f"Variable name '{var_name}' is not a valid Python identifier.")
        if not isinstance(var_value, (int, float)):
            raise ValueError(
                f"Variable '{var_name}' has non-numeric value: {var_value} "
                f"(type: {type(var_value).__name__})"
            )
        current_variable_values[var_name] = var_value

    try:
        # Parse the expression string into an AST (Abstract Syntax Tree)
        # 'eval' mode is used because we expect a single expression.
        ast_tree = ast.parse(math_expression, mode='eval')

        # Safely evaluate the AST
        result = _recursive_ast_eval(ast_tree, current_variable_values)

        # Ensure the result is a number before converting to string
        if not isinstance(result, (int, float)):
            # This should ideally not be reached if _recursive_ast_eval is correct
            raise ValueError("Expression did not evaluate to a numeric result.")
            
        return str(result)
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {e}")
    except ValueError: # Re-raise ValueErrors from _recursive_ast_eval or validation
        raise
    except Exception as e:
        # Catch any other unexpected errors during parsing or evaluation
        raise ValueError(f"Failed to evaluate expression: {e}")


if __name__ == '__main__':
    # Example Usage and Basic Tests
    test_expressions_no_vars = {
        "2 + 3": "5.0",
        "10 - 4": "6.0",
        "5 * 6": "30.0",
        "20 / 4": "5.0",
        "2 + 3 * 4": "14.0",
        "(2 + 3) * 4": "20.0",
        "10 / 2 - 1": "4.0",
        "-5 + 10": "5.0",
        "3 * -2": "-6.0",
        "1 / 3": str(1.0/3.0), # Ensure float division
        "  1 +   1  ": "2.0",
        "1.0": "1.0",
        "-3.14": "-3.14"
    }

    print("Running tests for evaluate_expression_safely (no variables):")
    for expr, expected in test_expressions_no_vars.items():
        try:
            result = evaluate_expression_safely(expr, {}) # Pass empty dict for no vars
            assert result == expected, f"Test failed for '{expr}': Expected {expected}, got {result}"
            print(f"PASS: '{expr}' -> '{result}'")
        except ValueError as e:
            print(f"FAIL (ValueError): '{expr}' -> {e}")
        except Exception as e:
            print(f"FAIL (Unexpected Error): '{expr}' -> {e}")

    print("\nRunning tests for evaluate_expression_safely with variables:")
    variable_test_cases = {
        ("x + y", {"x": 10, "y": 5}): "15.0",
        ("a - b", {"a": 10.5, "b": 3}): "7.5",
        ("var1 * var2", {"var1": 7, "var2": 6}): "42.0",
        ("total / count", {"total": 100, "count": 4}): "25.0",
        ("x + y * z", {"x": 1, "y": 2, "z": 3}): "7.0",
        ("(x + y) * z", {"x": 1, "y": 2, "z": 3}): "9.0",
        ("-val", {"val": 5}): "-5.0",
        ("val", {"val": 123.45}): "123.45",
        ("x", {"x": 0}): "0.0",
        ("num_1 / num_2", {"num_1": 1.0, "num_2": 2.0}): "0.5",
        ("var_int + var_float", {"var_int": 1, "var_float": 2.5}): "3.5",
    }
    for (expr, v_map), expected in variable_test_cases.items():
        try:
            result = evaluate_expression_safely(expr, v_map)
            assert result == expected, f"Test failed for '{expr}' with {v_map}: Expected {expected}, got {result}"
            print(f"PASS: '{expr}' with {v_map} -> '{result}'")
        except ValueError as e:
            print(f"FAIL (ValueError): '{expr}' with {v_map} -> {e}")
        except Exception as e:
            print(f"FAIL (Unexpected Error): '{expr}' with {v_map} -> {e}")

    print("\nRunning error case tests for evaluate_expression_safely:")
    error_test_inputs = [
        # Expression syntax/structure errors
        ("2 + ", {}, "Invalid syntax"),
        ("", {}, "Math expression cannot be empty."),
        ("   ", {}, "Math expression cannot be empty."),
        ("1 + (2 * 3", {}, "Invalid syntax"),
        ("1 + 2)", {}, "Invalid syntax"),
        ("1 + / 2", {}, "Invalid syntax"),
        # Unsupported operations/constructs
        ("2 ** 3", {}, "Unsupported binary operator: Pow"), # Assuming Pow is not in ALLOWED_OPERATORS
        ("import os", {}, "Invalid syntax"), # AST parse will fail
        ("eval('1+1')", {}, "Unsupported AST node type: Call"), # Or syntax error depending on AST strictness
        ("my_func()", {}, "Unsupported AST node type: Call"),
        # Variable-related errors
        ("x + 1", {}, "Undefined variable: 'x'"),
        ("y / 0", {"y": 10}, "Division by zero"),
        ("z * 2", {"z": "abc"}, "Variable 'z' has non-numeric value"),
        ("a + b", {"a": 1, "b": "text"}, "Variable 'b' has non-numeric value"),
        ("ok + 1", {123: 456}, "Variable name '123' must be a string."),
        ("x", {"x": "test"}, "Variable 'x' has non-numeric value"),
        ("x + y", {"x": 1}, "Undefined variable: 'y'"),
        ("1bad_var + 2", {}, "Invalid syntax"), # Invalid token for variable name
        ("var-name + 1", {}, "Invalid syntax"), # '-' in var name is subtraction
        ("with + 1", {}, "Invalid syntax"), # 'with' is a keyword
        ("valid_var", {"1invalid_key": 1}, "Variable name '1invalid_key' is not a valid Python identifier."),
        # Type errors for main arguments
        (123, {}, "Math expression must be a string."),
        ("1+1", "not a dict", "Variable mapping must be a dictionary."),
        ("1+1", [1,2,3], "Variable mapping must be a dictionary."),
    ]

    for expr_input, v_map_input, expected_msg_part in error_test_inputs:
        try:
            evaluate_expression_safely(expr_input, v_map_input)
            print(f"FAIL (Error not raised): For expression '{expr_input}' with map {v_map_input}")
        except ValueError as e:
            if expected_msg_part.lower() in str(e).lower():
                print(f"PASS (ValueError as expected): '{expr_input}' with {v_map_input} -> {e}")
            else:
                print(f"FAIL (Msg mismatch): '{expr_input}' with {v_map_input}. Expected part: '{expected_msg_part}', Got: '{e}'")
        except Exception as e: # Catch any other unexpected errors
            print(f"FAIL (Unexpected Exception type): '{expr_input}' with {v_map_input} -> {type(e).__name__}: {e}")
