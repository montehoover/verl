import ast

# Base list of allowed AST node types that are common across Python versions
# and represent the structure of arithmetic expressions.
_ALLOWED_NODE_TYPES_LIST = [
    ast.Expression,  # The root node of an expression parsed with mode='eval'.
    ast.BinOp,       # For binary operations like +, -, *, /.
    ast.UnaryOp,     # For unary operations like - (negation) or + (unary plus).
    ast.Add,         # The addition operator type.
    ast.Sub,         # The subtraction operator type.
    ast.Mult,        # The multiplication operator type.
    ast.Div,         # The division operator type.
    ast.UAdd,        # The unary plus operator type.
    ast.USub,        # The unary minus operator type.
]

# Add AST node types for numeric literals, handling Python version differences.
# Python 3.8+ uses ast.Constant for numbers, strings, None, True, False.
# Python < 3.8 uses ast.Num for numbers and ast.Constant for None, True, False.
# ast.NameConstant was used for None, True, False in Python < 3.8, and ast.Constant was an alias in 3.6, 3.7.
# To be precise and cover common versions (e.g., 3.6 to latest):
if hasattr(ast, 'Constant'):
    # ast.Constant exists in Python 3.6+.
    # In 3.6/3.7, it's for None/True/False.
    # In 3.8+, it's for numbers, strings, bytes, None, True, False.
    _ALLOWED_NODE_TYPES_LIST.append(ast.Constant)
if hasattr(ast, 'Num'):
    # ast.Num exists in Python <= 3.7 for numbers.
    # In Python 3.8, ast.Num is an alias for ast.Constant.
    # ast.Num is removed in Python 3.9+.
    _ALLOWED_NODE_TYPES_LIST.append(ast.Num)

# Convert the list to a tuple for efficient use with isinstance().
_ALLOWED_NODE_TYPES_TUPLE = tuple(set(_ALLOWED_NODE_TYPES_LIST)) # Use set to remove duplicates if any (e.g. ast.Num alias in 3.8)


def is_safe_expression(expression: str) -> bool:
    """
    Checks if the given arithmetic expression string is safe to evaluate.

    A safe expression can only contain:
    - Numbers (integers and floating-point).
    - Parentheses for grouping.
    - The basic arithmetic operations: addition (+), subtraction (-),
      multiplication (*), and division (/).
    - Unary plus (+) and unary minus (-).

    The function parses the expression into an Abstract Syntax Tree (AST)
    and verifies that all nodes in the tree correspond to allowed elements.
    This helps prevent evaluation of potentially harmful code, such as
    function calls, variable names, or other complex Python constructs.

    Args:
        expression: The user-provided arithmetic expression string.

    Returns:
        True if the expression is deemed safe, False otherwise.
        False is also returned for invalid syntax or non-string inputs.
    """
    if not isinstance(expression, str):
        return False  # Input must be a string.
    
    if not expression.strip():
        return False  # Empty or whitespace-only strings are not valid expressions.

    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError:
        return False  # The expression has invalid Python syntax.
    except ValueError: # Handles null bytes in expression
        return False

    for node in ast.walk(tree):
        # Check if the node type is in our whitelist of allowed types.
        if not isinstance(node, _ALLOWED_NODE_TYPES_TUPLE):
            return False  # Disallowed node type found.

        # Additional checks for specific node types:
        if hasattr(ast, 'Constant') and isinstance(node, ast.Constant):
            # If ast.Constant is used (Python 3.6+), ensure its value is a number.
            # This prevents evaluation of strings, None, True, False if they were
            # to be represented by ast.Constant in some Python version/context.
            if not isinstance(node.value, (int, float)):
                return False  # ast.Constant holding non-numeric data.
        
        # No specific check needed for ast.Num, as it inherently represents a number
        # in Python versions where it's distinct from ast.Constant (i.e., < 3.8).
        # If ast.Num is an alias for ast.Constant (Python 3.8), the ast.Constant check above applies.

        # Note: The operator types (ast.Add, ast.Sub, etc.) are themselves nodes
        # and are included in _ALLOWED_NODE_TYPES_TUPLE.
        # ast.BinOp and ast.UnaryOp nodes have an 'op' attribute which is an instance
        # of these operator types. ast.walk visits these 'op' nodes too.

    return True  # All nodes are of allowed types and satisfy specific checks.

if __name__ == '__main__':
    # Example Usage and Test Cases
    safe_expressions = [
        "1 + 2",
        "10 - 3.5",
        "2 * (3 + 4)",
        "100 / 5",
        "-5",
        "+3.14",
        "((1 + 2) * 3 - (4 / 2))",
        "1.0 * 2.0 / 3.0 + 4.0 - 5.0",
        "42",
        "  1 + 1  ", # Whitespace around
    ]

    unsafe_expressions = [
        "__import__('os').system('clear')", # Malicious code
        "print('hello')",                   # Function call
        "x + 1",                            # Variable name
        "1 ** 2",                           # Power operator (not allowed)
        "1 % 2",                            # Modulo operator (not allowed)
        "eval('1+1')",                      # eval itself
        "a.b",                              # Attribute access
        "[1, 2, 3]",                        # List literal
        "{'a': 1}",                         # Dict literal
        "lambda x: x + 1",                  # Lambda function
        "1 if True else 0",                 # Conditional expression
        "'hello' + 'world'",                # String concatenation
        "True",                             # Boolean literal
        "None",                             # None literal
        "",                                 # Empty string
        "   ",                              # Whitespace only string
        "1 + ",                             # Incomplete expression (SyntaxError)
        "1 + * 2",                          # Invalid syntax (SyntaxError)
        "1 + (2 * 3",                       # Mismatched parentheses (SyntaxError)
        "import os",                        # Import statement (not 'eval' mode, but good to list)
    ]

    print("Testing safe expressions:")
    for expr in safe_expressions:
        print(f"'{expr}': {is_safe_expression(expr)}")

    print("\nTesting unsafe expressions:")
    for expr in unsafe_expressions:
        print(f"'{expr}': {is_safe_expression(expr)}")

    # Test non-string input
    print("\nTesting non-string input:")
    print(f"123 (int): {is_safe_expression(123)}")
    print(f"None: {is_safe_expression(None)}")


def evaluate_expression(expression: str) -> str:
    """
    Evaluates a validated mathematical expression string and returns the result.

    The expression must first pass the `is_safe_expression` check.
    This function uses `eval()` to compute the result of the arithmetic expression.
    It handles specific arithmetic errors like division by zero.

    Args:
        expression: The validated arithmetic expression string.

    Returns:
        The computed result as a string.

    Raises:
        ValueError: If the expression is not safe, if a division by zero occurs,
                    or if another arithmetic error occurs during evaluation.
    """
    if not is_safe_expression(expression):
        raise ValueError("Invalid or unsafe expression provided.")

    try:
        # Use a restricted environment for eval.
        # Globals are empty, and builtins are explicitly emptied.
        result = eval(expression, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        raise ValueError("Error: Division by zero.")
    except ArithmeticError as e:
        # This can catch other math-related errors like OverflowError,
        # though Python's arbitrary precision integers and float 'inf'
        # handle many cases without raising exceptions.
        raise ValueError(f"Arithmetic error during evaluation: {str(e)}")
    # Other exceptions are not expected if is_safe_expression is robust,
    # as it should only allow basic arithmetic that results in numbers or
    # the above specific arithmetic errors.

    return str(result)


def exec_calculation(calc_string: str) -> str:
    """
    Processes a user-provided calculation string.

    It first checks if the expression is safe. If safe, it evaluates the
    expression. If the expression is unsafe or an error occurs during
    evaluation (e.g., division by zero), it returns 'Computation Error!'.
    Otherwise, it returns the computed result as a string.

    Args:
        calc_string: The user-provided arithmetic expression string.

    Returns:
        The computed result as a string, or 'Computation Error!' if the
        expression is unsafe or an evaluation error occurs.
    """
    if not is_safe_expression(calc_string):
        return "Computation Error!"
    
    try:
        result = evaluate_expression(calc_string)
        return result
    except ValueError: # Catches errors from evaluate_expression (e.g. div by zero, or re-check of safety)
        return "Computation Error!"

if __name__ == '__main__':
    # Example Usage and Test Cases for is_safe_expression (existing)
    safe_expressions = [
        "1 + 2",
        "10 - 3.5",
        "2 * (3 + 4)",
        "100 / 5",
        "-5",
        "+3.14",
        "((1 + 2) * 3 - (4 / 2))",
        "1.0 * 2.0 / 3.0 + 4.0 - 5.0",
        "42",
        "  1 + 1  ", # Whitespace around
    ]

    unsafe_expressions = [
        "__import__('os').system('clear')", # Malicious code
        "print('hello')",                   # Function call
        "x + 1",                            # Variable name
        "1 ** 2",                           # Power operator (not allowed)
        "1 % 2",                            # Modulo operator (not allowed)
        "eval('1+1')",                      # eval itself
        "a.b",                              # Attribute access
        "[1, 2, 3]",                        # List literal
        "{'a': 1}",                         # Dict literal
        "lambda x: x + 1",                  # Lambda function
        "1 if True else 0",                 # Conditional expression
        "'hello' + 'world'",                # String concatenation
        "True",                             # Boolean literal
        "None",                             # None literal
        "",                                 # Empty string
        "   ",                              # Whitespace only string
        "1 + ",                             # Incomplete expression (SyntaxError)
        "1 + * 2",                          # Invalid syntax (SyntaxError)
        "1 + (2 * 3",                       # Mismatched parentheses (SyntaxError)
        "import os",                        # Import statement (not 'eval' mode, but good to list)
    ]

    print("Testing safe expressions:")
    for expr in safe_expressions:
        print(f"'{expr}': {is_safe_expression(expr)}")

    print("\nTesting unsafe expressions:")
    for expr in unsafe_expressions:
        print(f"'{expr}': {is_safe_expression(expr)}")

    # Test non-string input for is_safe_expression (existing)
    print("\nTesting non-string input for is_safe_expression:")
    print(f"123 (int): {is_safe_expression(123)}")
    print(f"None: {is_safe_expression(None)}")

    # Test cases for evaluate_expression
    print("\nTesting evaluate_expression (successful cases):")
    eval_safe_expressions_data = {
        "1 + 2": "3",
        "10 - 3.5": "6.5",
        "2 * (3 + 4)": "14",
        "100 / 5": "20.0",
        "-5": "-5",
        "+3.14": "3.14",
        "((1 + 2) * 3 - (4 / 2))": "7.0",
        "1.0 * 2.0 / 3.0 + 4.0 - 5.0": str(1.0 * 2.0 / 3.0 + 4.0 - 5.0), # Use actual computation for expected float
        "42": "42",
        "0 / 5": "0.0",
        "100000000000000000000 * 2": "200000000000000000000",
        "1e10 * 2": "20000000000.0", # Float result
        "0.1 + 0.2": str(0.1 + 0.2), # Standard float precision example
    }
    for expr, expected in eval_safe_expressions_data.items():
        try:
            result = evaluate_expression(expr)
            # For float comparisons, ensure expected string matches Python's str() output
            match = (result == expected)
            if not match and '.' in expected: # Crude check if it might be a float precision issue
                try:
                    if abs(float(result) - float(expected)) < 1e-9: # Tolerate small float differences
                        match = True
                except ValueError:
                    pass # Not floats, or not convertible
            print(f"'{expr}' -> '{result}' (Expected: '{expected}', Match: {match})")
            if not match:
                 print(f"    Note: Python's float for '{expr}' is {eval(expr)}")
        except ValueError as e:
            print(f"'{expr}' -> Error: {e} (UNEXPECTED)")

    print("\nTesting evaluate_expression (expected error cases):")
    eval_error_expressions_data = [
        # Unsafe expressions (caught by is_safe_expression)
        ("x + 1", "Invalid or unsafe expression provided."),
        ("__import__('os').system('clear')", "Invalid or unsafe expression provided."),
        ("print('hello')", "Invalid or unsafe expression provided."),
        ("'abc' + 'def'", "Invalid or unsafe expression provided."),

        # Arithmetic errors
        ("1 / 0", "Error: Division by zero."),
        ("100 / 0", "Error: Division by zero."),
        ("1.0 / 0.0", "Error: Division by zero."),
        ("(5 + 5) / (2 - 2)", "Error: Division by zero."),
        ("1 / (1 - 1)", "Error: Division by zero."),
    ]

    for expr, expected_msg_part in eval_error_expressions_data:
        try:
            result = evaluate_expression(expr)
            print(f"'{expr}' -> '{result}' (UNEXPECTED SUCCESS)")
        except ValueError as e:
            if expected_msg_part in str(e):
                print(f"'{expr}' -> Error: {e} (EXPECTED)")
            else:
                print(f"'{expr}' -> Error: {e} (UNEXPECTED error message, expected part: '{expected_msg_part}')")
        except Exception as e:
            print(f"'{expr}' -> UNEXPECTED Exception type: {type(e).__name__}, Error: {e}")

    # Test cases for exec_calculation
    print("\nTesting exec_calculation:")
    exec_test_cases = {
        # Safe and valid expressions
        "1 + 2": "3",
        "10 - 3.5": "6.5",
        "2 * (3 + 4)": "14",
        "100 / 5": "20.0",
        "-5": "-5",
        "0 / 5": "0.0",
        "  1 + 1  ": "2", # With whitespace

        # Unsafe expressions
        "x + 1": "Computation Error!",
        "__import__('os').system('clear')": "Computation Error!",
        "print('hello')": "Computation Error!",
        "'abc' + 'def'": "Computation Error!",
        "1 ** 2": "Computation Error!",

        # Expressions leading to evaluation errors
        "1 / 0": "Computation Error!",
        "(5 + 5) / (2 - 2)": "Computation Error!",

        # Invalid syntax (caught by is_safe_expression)
        "1 +": "Computation Error!",
        "1 + * 2": "Computation Error!",
        "": "Computation Error!", # Empty string
        "   ": "Computation Error!", # Whitespace only string

        # Non-string input (is_safe_expression handles this, exec_calculation expects string)
        # For direct calls to exec_calculation, it should be a string.
        # If a non-string were passed, is_safe_expression would return False.
    }

    for expr, expected_output in exec_test_cases.items():
        actual_output = exec_calculation(expr)
        print(f"exec_calculation('{expr}') -> '{actual_output}' (Expected: '{expected_output}', Match: {actual_output == expected_output})")

    # Example of non-string input (though type hint suggests str)
    # This demonstrates that is_safe_expression within exec_calculation handles it.
    # However, static analysis would flag this as incorrect usage.
    # print(f"exec_calculation(123) -> '{exec_calculation(123)}' (Expected: 'Computation Error!', Match: {exec_calculation(123) == 'Computation Error!'})")
