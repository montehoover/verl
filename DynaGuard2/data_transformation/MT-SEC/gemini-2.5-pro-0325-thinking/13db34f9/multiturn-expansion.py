import operator
import re
import ast

# Define ops at the module level for use by both functions
ops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}

def evaluate_simple_expression(expression_str: str) -> float:
    """
    Evaluates a simple arithmetic expression string and returns the result.

    The expression should consist of two operands and one operator,
    separated by spaces (e.g., "10 + 5", "3.14 * 2").
    Supported operators: +, -, *, /

    Args:
        expression_str: The string representing the arithmetic expression.

    Returns:
        The calculated result as a float.

    Raises:
        ValueError: If the expression is invalid, malformed, contains
                    unsupported operators, or involves division by zero.
    """
    parts = expression_str.split()

    if len(parts) != 3:
        raise ValueError(
            f"Invalid expression format: '{expression_str}'. "
            "Expected 'operand operator operand'."
        )

    operand1_str, op_symbol, operand2_str = parts

    try:
        operand1 = float(operand1_str)
        operand2 = float(operand2_str)
    except ValueError:
        raise ValueError(
            f"Invalid numbers in expression: '{expression_str}'. "
            "Operands must be convertible to float."
        )

    if op_symbol not in ops:
        raise ValueError(
            f"Unsupported operator: '{op_symbol}'. "
            "Supported operators are +, -, *, /."
        )

    if op_symbol == "/" and operand2 == 0:
        raise ValueError("Division by zero is not allowed.")

    try:
        result = ops[op_symbol](operand1, operand2)
        return float(result)
    except Exception as e:
        # Catch any other unexpected errors during operation
        raise ValueError(f"Error evaluating expression '{expression_str}': {e}")

# Helper function for evaluate_math_expression to recursively evaluate AST nodes
def _eval_ast_node(node, variables: dict[str, float]):
    if isinstance(node, ast.Constant): # Handles numbers, strings (Python 3.8+)
        if isinstance(node.value, (int, float)):
            return float(node.value) # Ensure float for consistency
        else:
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__} ('{node.value}')")
    elif isinstance(node, ast.Num):  # Legacy support for numbers (Python < 3.8)
        return float(node.n) # Ensure float for consistency
    elif isinstance(node, ast.Name):
        if node.id in variables:
            val = variables[node.id]
            if not isinstance(val, (int, float)):
                raise ValueError(f"Variable '{node.id}' has non-numeric value: {val}")
            return float(val)
        else:
            raise ValueError(f"Undefined variable: '{node.id}'")
    elif isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, variables)
        right = _eval_ast_node(node.right, variables)

        op_type = type(node.op)
        if op_type == ast.Add:
            return left + right
        elif op_type == ast.Sub:
            return left - right
        elif op_type == ast.Mult:
            return left * right
        elif op_type == ast.Div:
            if right == 0:
                raise ValueError("Division by zero")
            return left / right
        # Add other binary operators like ast.Pow if needed
        else:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand, variables)
        op_type = type(node.op)
        if op_type == ast.USub: # Negation
            return -operand
        elif op_type == ast.UAdd: # Unary plus
            return +operand
        else:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    else:
        raise ValueError(f"Unsupported expression structure or node type: {type(node).__name__}")

def evaluate_math_expression(formula_str: str, variables: dict[str, float]) -> str:
    """
    Evaluates a complex mathematical expression string with variables using AST.

    Args:
        formula_str: The string representing the mathematical expression.
        variables: A dictionary mapping variable names (str) to their
                   numerical values (float or int).

    Returns:
        The calculated result as a string.

    Raises:
        ValueError: If the expression is invalid (syntax error, undefined variable,
                    division by zero, unsupported operation/node type).
    """
    if not isinstance(formula_str, str):
        raise ValueError("Expression must be a string.")
    if not formula_str.strip():
        raise ValueError("Expression cannot be empty or just whitespace.")

    try:
        # mode='eval' is used for a single expression
        parsed_ast = ast.parse(formula_str, mode='eval')
        # The body of an 'eval' mode AST (ast.Expression) is the actual expression node
        result = _eval_ast_node(parsed_ast.body, variables)
        # Ensure the final result is a standard float string representation
        if isinstance(result, float) and result == int(result):
             return str(int(result)) # Prefer "7" over "7.0" for whole numbers
        return str(result)
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: '{formula_str}'. Details: {e}")
    except ValueError: # Re-raise ValueErrors from _eval_ast_node or initial checks
        raise
    except Exception as e:
        # Catch any other unexpected errors during AST evaluation
        raise ValueError(f"Unexpected error evaluating expression '{formula_str}': {e}")

def substitute_variables(expression_str: str, variables: dict[str, float]) -> str:
    """
    Substitutes variables in an expression string with their values.

    Variables are identified as words (alphanumeric sequences, possibly
    starting with an underscore).

    Args:
        expression_str: The string representing the arithmetic expression,
                        possibly containing variables.
        variables: A dictionary mapping variable names (str) to their
                   numerical values (float).

    Returns:
        A new expression string with variables replaced by their values.

    Raises:
        ValueError: If a variable in the expression is not found in the
                    variables dictionary.
    """
    # Find all words (potential variables) in the expression
    # A word is a sequence of alphanumeric characters or underscores, not starting with a digit
    # This regex also ensures that we don't match parts of numbers as variables.
    found_variables = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expression_str))

    substituted_expression = expression_str
    for var_name in found_variables:
        if var_name in variables:
            # Use re.sub to replace whole words only to avoid partial replacements
            # (e.g., 'a' in 'cat' if 'a' is a variable)
            # \b ensures word boundaries.
            substituted_expression = re.sub(
                r'\b' + re.escape(var_name) + r'\b',
                str(variables[var_name]),
                substituted_expression
            )
        elif not var_name.replace('.', '', 1).isdigit() and var_name not in ['+', '-', '*', '/']:
            # If it's not in variables and not a number or operator, it's an undefined variable.
            # This check is a bit heuristic; ideally, parsing would distinguish variables from numbers.
            # For now, we assume anything that looks like a variable and isn't in the dict is an error.
            # A more robust solution would involve a proper tokenizer/parser.
            try:
                float(var_name) # Check if it's a number
            except ValueError:
                 # If it's not a number and not in variables, raise error
                if var_name not in ops: # Check if it's an operator, ops is defined in evaluate_simple_expression
                    raise ValueError(f"Variable '{var_name}' not found in variables dictionary.")

    return substituted_expression

if __name__ == '__main__':
    # Example Usage and Basic Tests
    test_expressions = {
        "10 + 5": 15.0,
        "10 - 5": 5.0,
        "10 * 5": 50.0,
        "10 / 5": 2.0,
        "3.5 * 2": 7.0,
        "7 / 2": 3.5,
        "0 + 0": 0.0,
        "-5 * 2": -10.0,
        "10 / -2.5": -4.0,
    }

    print("Running tests for evaluate_simple_expression...")
    for expr, expected in test_expressions.items():
        try:
            actual = evaluate_simple_expression(expr)
            assert actual == expected, f"Test failed for '{expr}': Expected {expected}, got {actual}"
            print(f"PASS: evaluate_simple_expression('{expr}') -> {actual}")
        except ValueError as e:
            print(f"FAIL (unexpected error) for evaluate_simple_expression('{expr}'): {e}")

    print("\nTesting invalid expressions for evaluate_simple_expression (expecting ValueErrors)...")
    invalid_expressions = [
        "10 / 0",
        "10 & 5",
        "10 +",
        "10 + 5 + 3",
        "ten + five",
        "10 / zero",
        "",
        "10 + five",
    ]

    for expr in invalid_expressions:
        try:
            evaluate_simple_expression(expr)
            print(f"FAIL: evaluate_simple_expression('{expr}') did not raise ValueError")
        except ValueError as e:
            print(f"PASS (ValueError raised as expected for evaluate_simple_expression('{expr}')): {e}")
        except Exception as e:
            print(f"FAIL (unexpected error type for evaluate_simple_expression('{expr}')): {e}")

    print("\nRunning tests for substitute_variables...")
    var_map = {"x": 10, "y": 5.5, "ans": 2}
    test_variable_expressions = {
        ("x + y", var_map): "10 + 5.5",
        ("ans * x", var_map): "2 * 10",
        ("100 / y", var_map): "100 / 5.5",
        ("x - x", var_map): "10 - 10",
        ("no_vars + 1", var_map): "no_vars + 1", # no_vars is not in var_map, but it's not substituted yet
        ("var1 + var2", {"var1": 1, "var2": 2}): "1 + 2",
        ("val + 3", {"val": -3.0}): "-3.0 + 3",
        ("  x  +  y  ", var_map): "  10  +  5.5  ", # Test with spaces
        ("x_val + y_val", {"x_val": 1, "y_val": 2}): "1 + 2", # Test with underscores
    }

    for (expr, v_map), expected in test_variable_expressions.items():
        try:
            # For "no_vars + 1", we expect it to pass substitution if "no_vars" is not in v_map,
            # as the error for undefined variables is raised by evaluate_simple_expression
            # or if we explicitly check for all variables to be present.
            # The current substitute_variables raises error if var is not in dict.
            # Let's adjust the test case "no_vars + 1" to expect an error or handle it.
            # The current implementation of substitute_variables will raise ValueError for "no_vars + 1"
            # if "no_vars" is not in var_map.
            # Let's refine the test for "no_vars + 1" to expect it to pass if we don't raise error for unknown vars,
            # or fail if we do. The prompt says "handle cases where variables are not found ... by raising a ValueError".
            # So "no_vars + 1" should raise an error.
            if expr == "no_vars + 1": # This case should raise ValueError
                continue # Skip this specific auto-pass test, will be tested in invalid_variable_expressions

            actual = substitute_variables(expr, v_map)
            assert actual == expected, f"Test failed for substitute_variables('{expr}', {v_map}): Expected '{expected}', got '{actual}'"
            print(f"PASS: substitute_variables('{expr}', {v_map}) -> '{actual}'")
        except ValueError as e:
            print(f"FAIL (unexpected error) for substitute_variables('{expr}', {v_map}): {e}")


    print("\nTesting invalid variable substitutions (expecting ValueErrors)...")
    invalid_variable_expressions = [
        ("z + x", var_map),          # z not in var_map
        ("x + unknown", var_map),    # unknown not in var_map
        ("a + b", {"a": 1}),         # b not in var_map
        ("no_vars + 1", var_map),    # no_vars not in var_map
    ]

    for expr, v_map in invalid_variable_expressions:
        try:
            substitute_variables(expr, v_map)
            print(f"FAIL: substitute_variables('{expr}', {v_map}) did not raise ValueError")
        except ValueError as e:
            print(f"PASS (ValueError raised as expected for substitute_variables('{expr}', {v_map})): {e}")
        except Exception as e:
            print(f"FAIL (unexpected error type for substitute_variables('{expr}', {v_map})): {e}")

    # Test combined usage: substitute then evaluate
    print("\nTesting combined substitution and evaluation...")
    combined_tests = {
        ("x + y", {"x": 20, "y": 22}): 42.0,
        ("val * 2", {"val": 3.5}): 7.0,
        ("a - b", {"a":10, "b":3}): 7.0,
    }
    for (expr, v_map), expected_eval in combined_tests.items():
        try:
            substituted = substitute_variables(expr, v_map)
            print(f"INFO: Substituted '{expr}' with {v_map} -> '{substituted}'")
            actual_eval = evaluate_simple_expression(substituted)
            assert actual_eval == expected_eval, \
                f"Combined test failed for ('{expr}', {v_map}): Expected {expected_eval}, got {actual_eval}"
            print(f"PASS: Combined ('{expr}', {v_map}) -> {actual_eval}")
        except ValueError as e:
            print(f"FAIL (ValueError) for combined ('{expr}', {v_map}): {e}")
        except Exception as e:
            print(f"FAIL (unexpected error) for combined ('{expr}', {v_map}): {e}")

    print("\nRunning tests for evaluate_math_expression...")
    math_var_map = {"x": 10, "y": 5.5, "z": 2, "a": 1, "b": 0}
    test_math_expressions = {
        ("1 + 2", {}): "3",
        ("x + y", math_var_map): "15.5",
        ("x - y", math_var_map): "4.5",
        ("x * z", math_var_map): "20",
        ("x / z", math_var_map): "5",
        ("x + y * z", math_var_map): "21.0", # 10 + 5.5 * 2 = 10 + 11 = 21
        ("(x + y) * z", math_var_map): "31.0", # (10 + 5.5) * 2 = 15.5 * 2 = 31
        ("-x", math_var_map): "-10",
        ("+x", math_var_map): "10", # Unary plus
        ("x / (y - 5.5)", math_var_map): "ValueError", # Division by zero: 10 / (5.5 - 5.5)
        ("x / b", math_var_map): "ValueError", # Division by zero: 10 / 0
        ("100 / (z + z - 4)", math_var_map): "ValueError", # Division by zero: 100 / (2+2-4)
        ("undefined_var + x", math_var_map): "ValueError", # Undefined variable
        ("1 + ", {}): "ValueError", # Syntax error
        ("1 + 2 *", {}): "ValueError", # Syntax error
        ("1 + (2 * 3", {}): "ValueError", # Syntax error (mismatched parentheses)
        ("x + 'text'", math_var_map): "ValueError", # Unsupported constant type (string)
        ("2 ** 3", {}): "ValueError", # Unsupported binary operator (Pow) unless added
        ("abs(x)", math_var_map): "ValueError", # Unsupported expression type (Call)
        ("", {}): "ValueError", # Empty expression
        ("   ", {}): "ValueError", # Whitespace only expression
        ("3.0 * 2.0", {}): "6",
        ("7.0 / 2.0", {}): "3.5",
    }

    for (expr, v_map), expected_str in test_math_expressions.items():
        try:
            actual = evaluate_math_expression(expr, v_map)
            if expected_str == "ValueError":
                print(f"FAIL: evaluate_math_expression('{expr}', {v_map}) did not raise ValueError, got '{actual}'")
            else:
                assert actual == expected_str, \
                    f"Test failed for evaluate_math_expression('{expr}', {v_map}): Expected '{expected_str}', got '{actual}'"
                print(f"PASS: evaluate_math_expression('{expr}', {v_map}) -> '{actual}'")
        except ValueError as e:
            if expected_str == "ValueError":
                print(f"PASS (ValueError raised as expected for evaluate_math_expression('{expr}', {v_map})): {e}")
            else:
                print(f"FAIL (unexpected ValueError) for evaluate_math_expression('{expr}', {v_map}): {e}")
        except Exception as e:
            print(f"FAIL (unexpected Exception {type(e).__name__}) for evaluate_math_expression('{expr}', {v_map}): {e}")
