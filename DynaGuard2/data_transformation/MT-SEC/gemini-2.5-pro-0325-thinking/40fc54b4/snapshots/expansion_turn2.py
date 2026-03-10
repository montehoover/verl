import ast

def _is_allowed_node(node):
    """
    Recursively checks if an AST node is allowed in a basic arithmetic expression.
    Allowed nodes: Numbers (int, float), BinOp (+, -, *, /), UnaryOp (USub, UAdd).
    """
    if isinstance(node, ast.Expression):
        # The root of an 'eval' mode parse is an Expression node.
        # We need to check its body.
        return _is_allowed_node(node.body)
    elif isinstance(node, ast.Constant):
        # For Python 3.8+, numbers (and other literals) are ast.Constant.
        # We only allow numeric constants.
        return isinstance(node.value, (int, float))
    elif isinstance(node, ast.Num):
        # For Python < 3.8, numbers are ast.Num.
        # This is included for broader compatibility, though ast.Constant is preferred.
        return True  # ast.Num always contains a number
    elif isinstance(node, ast.BinOp):
        # Binary operations: +, -, *, /
        allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div)
        if not isinstance(node.op, allowed_bin_ops):
            return False
        # Recursively check left and right operands
        return _is_allowed_node(node.left) and _is_allowed_node(node.right)
    elif isinstance(node, ast.UnaryOp):
        # Unary operations: primarily unary minus (e.g., -5) and unary plus (e.g., +5)
        allowed_unary_ops = (ast.USub, ast.UAdd)
        if not isinstance(node.op, allowed_unary_ops):
            return False
        # Recursively check the operand
        return _is_allowed_node(node.operand)
    
    # Any other node type is not allowed (e.g., ast.Name, ast.Call, etc.)
    return False

def is_valid_expression(expression: str) -> bool:
    """
    Checks if a user-provided string is a well-formed arithmetic expression
    using only basic operations (addition, subtraction, multiplication, division)
    and numbers (integers or floats).

    Args:
        expression: The string input representing the arithmetic expression.

    Returns:
        True if the expression is valid, False otherwise.
    """
    if not isinstance(expression, str) or not expression.strip():
        # Expression must be a non-empty string.
        return False

    try:
        # Parse the expression string into an AST.
        # mode='eval' is used because we are parsing a single expression,
        # not a full module or statement.
        tree = ast.parse(expression, mode='eval')
        
        # Traverse the AST to ensure only allowed nodes and operations are present.
        return _is_allowed_node(tree)
    except SyntaxError:
        # If ast.parse raises a SyntaxError, the expression is not well-formed.
        return False
    except Exception:
        # Catch any other potential errors during parsing or AST traversal
        # (e.g., recursion depth exceeded for extremely complex expressions, though unlikely here).
        return False

def calculate_expression(expression: str):
    """
    Calculates the result of a validated mathematical expression string.

    Args:
        expression: The string input representing the arithmetic expression.

    Returns:
        The calculated result (int or float) if successful,
        or an error message string if the expression is invalid, unsafe,
        or results in a calculation error (e.g., division by zero).
    """
    if not is_valid_expression(expression):
        return "Error: Invalid or unsafe expression."

    try:
        # Using eval() here is considered safe because is_valid_expression
        # has already parsed the string and ensured it only contains
        # basic arithmetic operations and numbers.
        result = eval(expression)
        return result
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        # Catch any other calculation errors.
        return f"Error: Calculation failed. {str(e)}"

if __name__ == '__main__':
    # Example Usage and Basic Tests for is_valid_expression
    print("--- Testing is_valid_expression ---")
    test_expressions_validity = {
        "1 + 2": True,
        "3 * 5 - 2 / 4": True,
        "10": True,
        "-5": True,
        "+3.14": True,
        "2.5 * (4 + -1)": True,
        "(1 + 2) * 3": True,
        "1 / 0": True,  # Syntactically valid, division by zero is a runtime error
        "1 +": False,
        "* 2": False,
        "1 + 2 *": False,
        "1 + foo": False,
        "abs(-5)": False,
        "1 + '2'": False, # String literals not allowed as numbers
        "": False,
        "   ": False, # Whitespace only is not valid
        "1 + (2 * 3))": False, # Mismatched parentheses
        "1 + (2 * 3": False,   # Mismatched parentheses
        "1 + + 2": False, # Typically not allowed by strict parsers, ast.parse might allow depending on context
                          # ast.parse("1++2") is valid, becomes 1 + (+2)
                          # ast.parse("1 + + 2") is valid.
                          # Our _is_allowed_node will handle this correctly.
                          # "1++2" -> BinOp(left=1, op=Add, right=UnaryOp(op=UAdd, operand=2)) -> True
        "1 + --2": True,  # Becomes 1 + (-(-2))
        "1 + * 2": False, # Consecutive binary operators
    }

    for expr, expected in test_expressions_validity.items():
        result = is_valid_expression(expr)
        print(f"Expression: \"{expr}\", Valid: {result}, Expected: {expected}, Match: {result == expected}")

    # Specific test for "1 + + 2" which ast.parse handles as 1 + (+2)
    # Our checker should allow this as UAdd is permitted.
    expr_plus_plus = "1 + +2" # Note: "1 + + 2" (with space) is also valid by ast.parse
    expected_validity = True
    result_validity = is_valid_expression(expr_plus_plus)
    print(f"Expression: \"{expr_plus_plus}\", Valid: {result_validity}, Expected: {expected_validity}, Match: {result_validity == expected_validity}")
    
    expr_plus_plus_space = "1 + + 2" # With space
    expected_validity_space = True
    result_validity_space = is_valid_expression(expr_plus_plus_space)
    print(f"Expression: \"{expr_plus_plus_space}\", Valid: {result_validity_space}, Expected: {expected_validity_space}, Match: {result_validity_space == expected_validity_space}")

    # Test for ast.Num for older Python if needed (current code uses ast.Constant)
    # If running on Python < 3.8, ast.parse("1") would yield ast.Num(n=1)
    # Our _is_allowed_node has a branch for ast.Num.
    # Example: Manually creating an ast.Num node to test that path (if possible and relevant)
    # This part is more for illustrating the ast.Num handling logic.
    try:
        # Test with a simple number, which would be ast.Num on older Pythons
        # or ast.Constant on newer.
        num_expr_tree = ast.parse("123", mode='eval')
        if isinstance(num_expr_tree.body, ast.Num):
            print("Testing with ast.Num node type (Python < 3.8 behavior for numbers)")
            assert _is_allowed_node(num_expr_tree) == True
        elif isinstance(num_expr_tree.body, ast.Constant) and isinstance(num_expr_tree.body.value, int):
            print("Testing with ast.Constant node type (Python >= 3.8 behavior for numbers)")
            assert _is_allowed_node(num_expr_tree) == True

    except Exception as e:
        print(f"Error during specific ast.Num/Constant test: {e}")

    print("\n--- Testing calculate_expression ---")
    test_calculations = {
        "1 + 2": 3,
        "3 * 5 - 2 / 4": 14.5,
        "10": 10,
        "-5": -5,
        "+3.14": 3.14,
        "2.5 * (4 + -1)": 7.5,
        "(1 + 2) * 3": 9,
        "1 / 0": "Error: Division by zero.",
        "1 +": "Error: Invalid or unsafe expression.",
        "* 2": "Error: Invalid or unsafe expression.",
        "1 + foo": "Error: Invalid or unsafe expression.",
        "abs(-5)": "Error: Invalid or unsafe expression.",
        "1 + +2": 3, # Valid: 1 + (+2)
        "1 + --2": 3, # Valid: 1 + (-(-2)) which is 1 + 2
        "10 / 2.0": 5.0,
        "10 / (2 + 3)": 2.0,
        "10 / (2 - 2)": "Error: Division by zero.",
        "2**3": "Error: Invalid or unsafe expression." # Exponentiation not allowed by _is_allowed_node
    }

    for expr, expected_result in test_calculations.items():
        actual_result = calculate_expression(expr)
        # Comparing floats requires tolerance, but for error strings exact match is fine.
        if isinstance(actual_result, float) and isinstance(expected_result, float):
            match = abs(actual_result - expected_result) < 1e-9
        else:
            match = actual_result == expected_result
        print(f"Expression: \"{expr}\", Result: {actual_result}, Expected: {expected_result}, Match: {match}")

    # Test case for an expression that is syntactically valid but uses disallowed operations
    # (e.g. if we were to disallow ast.UAdd, but current _is_allowed_node allows it)
    # For now, "1++2" is valid and calculates to 3.
    # If we wanted to make "1++2" invalid, _is_allowed_node would need adjustment.
    # Current behavior:
    expr_double_unary = "1++2" # ast.parse makes this 1 + (+2)
    expected_calc_double_unary = 3
    actual_calc_double_unary = calculate_expression(expr_double_unary)
    print(f"Expression: \"{expr_double_unary}\", Result: {actual_calc_double_unary}, Expected: {expected_calc_double_unary}, Match: {actual_calc_double_unary == expected_calc_double_unary}")
