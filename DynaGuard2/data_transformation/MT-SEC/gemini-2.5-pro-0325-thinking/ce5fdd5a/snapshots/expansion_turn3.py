import ast
import re
import sys

def is_valid_expression(expression_string: str) -> bool:
    """
    Checks if a string is a valid arithmetic expression.

    A valid expression contains only numbers (integers or floats),
    operators (+, -, *, /), and parentheses.
    It aims to prevent malicious inputs by not using eval() directly
    and by strictly validating the expression's components.

    Args:
        expression_string: The string to validate.

    Returns:
        True if the expression is valid, False otherwise.
    """
    if not isinstance(expression_string, str):
        return False

    stripped_expression = expression_string.strip()
    if not stripped_expression:
        return False

    # 1. Character validation: Allow only digits, operators, parentheses, dots, and whitespace.
    # This is a preliminary check; ast.parse will do more thorough syntax validation.
    allowed_chars_pattern = r"^[0-9\s\+\-\*\/\(\)\.]*$"
    if not re.fullmatch(allowed_chars_pattern, stripped_expression):
        return False

    # 2. Syntax validation: Try to parse the expression using ast.parse.
    # This will catch syntax errors like unbalanced parentheses, invalid operators, etc.
    try:
        tree = ast.parse(stripped_expression, mode='eval')
    except SyntaxError:
        return False
    except Exception: # Catch any other parsing related errors
        return False


    # 3. AST node validation: Walk through the AST to ensure only allowed node types and operations are present.
    # This prevents constructs like function calls, variable names, attribute access, etc.
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression):
            # This is the root node for expressions parsed in 'eval' mode.
            pass
        elif sys.version_info >= (3, 8) and isinstance(node, ast.Constant):
            # For Python 3.8+, ast.Constant is used for literals (numbers, strings, None, bools).
            # We only allow numeric constants (int or float).
            if not isinstance(node.value, (int, float)):
                return False
        elif sys.version_info < (3, 8) and isinstance(node, ast.Num):
            # For Python < 3.8, ast.Num is used for numeric literals.
            # ast.Num.n is always an int or float, so no further type check on n is strictly needed here.
            pass
        elif isinstance(node, ast.BinOp):
            # Binary operations: check if the operator is one of the allowed ones (+, -, *, /).
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                return False
        elif isinstance(node, ast.UnaryOp):
            # Unary operations: check if the operator is one of the allowed ones (+, -).
            # e.g., -5 or +5
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                return False
        # Disallow any other AST node types. This includes (but is not limited to):
        # ast.Name (variables), ast.Call (function calls), ast.Attribute (attribute access),
        # ast.Compare (e.g., x < 5), ast.BoolOp (and, or),
        # ast.List, ast.Tuple, ast.Dict, ast.Set,
        # ast.Subscript, ast.Slice,
        # String constants (ast.Str for Python < 3.8, or ast.Constant(value="...") for 3.8+),
        # Boolean/None constants (ast.NameConstant for < 3.8, or ast.Constant(value=True/False/None) for 3.8+).
        # The checks for ast.Constant/ast.Num above handle numeric literals.
        # Any non-numeric literal or other construct will fall into this else clause.
        else:
            # If the node type is not explicitly allowed and handled above, reject the expression.
            # This is a catch-all for any disallowed constructs.
            # For Python < 3.8, if it's an ast.Str or ast.NameConstant, it will be caught here.
            # For Python 3.8+, if ast.Constant holds a string/bool/None, it's caught by the specific check.
            # This 'else' ensures that only the whitelisted node structures pass.
            if not (isinstance(node, ast.Load)): # ast.Load is a context, not a value/operation. It's fine.
                                                 # However, ast.Load appears with ast.Name, which we want to disallow.
                                                 # The current logic correctly disallows ast.Name.
                                                 # If an unknown node type that is not an issue appears, it might need to be added.
                                                 # For now, this strictness is safer.
                # Re-evaluating the 'else' condition:
                # The types explicitly checked and passed (Expression, Constant/Num, BinOp, UnaryOp)
                # are the only ones allowed. Any node not matching these (e.g., Name, Call, Attribute, Str, etc.)
                # will not match any of the `elif` conditions and thus implicitly fall through.
                # The loop should `continue` if a node is valid, and `return False` if invalid.
                # The current structure is:
                # if type A: pass/check_and_maybe_return_False
                # elif type B: pass/check_and_maybe_return_False
                # ...
                # else: return False (this is the key for unlisted types)
                # This means the current structure is:
                # if Expression: pass
                # elif Constant/Num: if not numeric, return False
                # elif BinOp: if op not allowed, return False
                # elif UnaryOp: if op not allowed, return False
                # else: return False (catches ast.Name, ast.Call, ast.Str, etc.)
                # This logic is correct. No change needed here.
                return False # Node type not in whitelist

    return True


def calculate_expression(expression_string: str) -> str:
    """
    Safely evaluates a pre-validated arithmetic expression string.

    Args:
        expression_string: The arithmetic expression string.
                           It's assumed this string has already passed
                           is_valid_expression().

    Returns:
        The result of the calculation as a string.
        Returns "Error: Division by zero" for division by zero.
        Returns "Error: Invalid expression" for other calculation errors
        or if the expression, despite prior validation, leads to an
        unhandled AST node during evaluation (should be rare).
    """
    if not is_valid_expression(expression_string):
        # This is a safeguard, but the caller should ensure validity.
        return "Error: Invalid expression provided to calculate_expression"

    try:
        tree = ast.parse(expression_string.strip(), mode='eval')
    except Exception:
        # Should not happen if is_valid_expression passed
        return "Error: Could not parse expression"

    def _evaluate_node(node):
        if isinstance(node, ast.Expression):
            return _evaluate_node(node.body)
        elif sys.version_info >= (3, 8) and isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                # This case should be caught by is_valid_expression
                raise TypeError("Unsupported constant type")
        elif sys.version_info < (3, 8) and isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            left_val = _evaluate_node(node.left)
            right_val = _evaluate_node(node.right)
            if isinstance(node.op, ast.Add):
                return left_val + right_val
            elif isinstance(node.op, ast.Sub):
                return left_val - right_val
            elif isinstance(node.op, ast.Mult):
                return left_val * right_val
            elif isinstance(node.op, ast.Div):
                if right_val == 0:
                    raise ZeroDivisionError("Division by zero")
                return left_val / right_val
            else:
                # This case should be caught by is_valid_expression
                raise TypeError("Unsupported binary operator")
        elif isinstance(node, ast.UnaryOp):
            operand_val = _evaluate_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand_val
            elif isinstance(node.op, ast.USub):
                return -operand_val
            else:
                # This case should be caught by is_valid_expression
                raise TypeError("Unsupported unary operator")
        else:
            # This case should be caught by is_valid_expression
            raise TypeError(f"Unsupported AST node type: {type(node)}")

    try:
        result = _evaluate_node(tree)
        # Format to avoid potential floating point inaccuracies in string representation
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except TypeError as e: # Catch errors from unsupported AST nodes/ops
        return f"Error: Invalid operation ({e})"
    except Exception: # Catch any other unexpected errors during evaluation
        return "Error: Calculation failed"


def evaluate_expression(expr: str) -> str:
    """
    Processes a user expression string, validates it, and safely evaluates it.

    Args:
        expr: The string input to interpret and compute.

    Returns:
        The result of the calculation as a string if the expression is valid
        and calculable, otherwise "Error!".
    """
    try:
        if not is_valid_expression(expr):
            return "Error!"

        # Expression is syntactically valid, now try to calculate it.
        # calculate_expression will handle specific calculation errors like division by zero.
        calculation_result = calculate_expression(expr)

        # If calculate_expression returned an error string, propagate as "Error!"
        if calculation_result.startswith("Error:"):
            return "Error!"

        return calculation_result
    except Exception:
        # Catch-all for any other unexpected issues during validation or calculation steps.
        return "Error!"


if __name__ == '__main__':
    # Test cases
    valid_expressions = [
        "1 + 1",
        "2 * (3 + 4)",
        "10 / 2",
        "5 - 1",
        "1.5 + 2.5",
        "-3 * 2",
        "+10 - -5",
        "((1 + 2) * 3) - (4 / (5 + 6))",
        "1",
        "-1",
        "+1.0",
        "  1 + 1  ", # Test with whitespace
        ".5 * 2", # Test leading dot float
        "5. * 2", # Test trailing dot float
    ]

    invalid_expressions = [
        "1 + ", # Incomplete expression
        "(1 + 2", # Unbalanced parentheses
        "1 + 2)", # Unbalanced parentheses
        "1 +* 2", # Invalid operator sequence
        "abc + 1", # Contains variables/names
        "print(1)", # Contains function call
        "1 + foo()", # Contains function call
        "1 + object.attribute", # Contains attribute access
        "1 + (lambda x: x)(1)", # Lambda function
        "1 < 2", # Comparison operator
        "True and False", # Boolean operators
        "1_000 + 1", # Underscores in numbers (allowed by Python but not by regex here)
        "1e3 + 1", # Scientific notation (allowed by Python but not by regex here)
        "\"hello\" + \"world\"", # String literals
        "eval('1+1')", # eval itself
        "", # Empty string
        "   ", # Whitespace only string
        "1.2.3", # Invalid number format
        "1 + [1, 2]", # List
        "1 + { 'a': 1 }", # Dictionary
        "1 + (1,)", # Tuple
        "1 % 2", # Modulo operator (not in +, -, *, /)
        "2 ** 3", # Power operator (not in +, -, *, /)
        "~1", # Bitwise NOT (not UAdd/USub)
    ]

    print("Valid expressions:")
    for expr in valid_expressions:
        if is_valid_expression(expr):
            print(f"  '{expr}' -> Valid (Correct)")
        else:
            print(f"  '{expr}' -> Invalid (INCORRECT!)")

    print("\nInvalid expressions:")
    for expr in invalid_expressions:
        if not is_valid_expression(expr):
            print(f"  '{expr}' -> Invalid (Correct)")
        else:
            print(f"  '{expr}' -> Valid (INCORRECT!)")

    # Specific check for non-string input
    print("\nNon-string input test:")
    if not is_valid_expression(123):
        print(f"  `123` (int) -> Invalid (Correct)")
    else:
        print(f"  `123` (int) -> Valid (INCORRECT!)")

    print("\nCalculation tests:")
    calculation_test_cases = {
        "1 + 1": "2",
        "2 * (3 + 4)": "14", # 2 * 7
        "10 / 2": "5",
        "5 - 1": "4",
        "1.5 + 2.5": "4", # 4.0 -> 4
        "-3 * 2": "-6",
        "+10 - -5": "15",
        "((1 + 2) * 3) - (4 / (5 + 6))": str(9 - (4/11.0)), # 9 - 0.3636...
        "1": "1",
        "-1": "-1",
        "+1.0": "1",
        "  1 + 1  ": "2",
        ".5 * 2": "1", # 1.0 -> 1
        "5. * 2": "10",
        "10 / 0": "Error: Division by zero",
        "10 / (2 - 2)": "Error: Division by zero",
        "1 / 3": str(1/3.0),
        "2.0 * 3.5": "7", # 7.0 -> 7
    }

    for expr, expected in calculation_test_cases.items():
        if is_valid_expression(expr):
            result = calculate_expression(expr)
            if result == expected:
                print(f"  Calculate '{expr}' -> '{result}' (Correct)")
            else:
                print(f"  Calculate '{expr}' -> '{result}' (Expected '{expected}', INCORRECT!)")
        else:
            # For expressions that are invalid, calculate_expression might return an error
            # This part of the test primarily focuses on valid expressions for calculation
            if expr == "10 / 0" or expr == "10 / (2 - 2)": # These are syntactically valid but cause runtime error
                 result = calculate_expression(expr)
                 if result == expected:
                    print(f"  Calculate '{expr}' (runtime error expected) -> '{result}' (Correct)")
                 else:
                    print(f"  Calculate '{expr}' (runtime error expected) -> '{result}' (Expected '{expected}', INCORRECT!)")
            else:
                print(f"  '{expr}' is invalid, skipping calculation test (or handle as error case if needed)")

    print("\nEvaluate expression tests:")
    evaluate_test_cases = {
        "1 + 1": "2",
        "2 * (3 + 4)": "14",
        "10 / 2": "5",
        "  (  1 + 1 ) * 2  ": "4",
        "10 / 0": "Error!", # Division by zero
        "1 + abc": "Error!", # Invalid characters/syntax
        "print(1)": "Error!", # Invalid construct
        "1 +": "Error!", # Incomplete expression
        "1.0 / 2.0": "0.5",
        "-5 + -5": "-10",
        "100": "100",
        "10 / (2-2)": "Error!", # Division by zero within parentheses
        "1 / 3.0": str(1/3.0),
        "": "Error!", # Empty string
        "   ": "Error!", # Whitespace only
        "1.2.3": "Error!", # Invalid number
        "1 + [1,2]": "Error!", # Invalid construct
        "1 % 2": "Error!", # Invalid operator
    }

    for expr, expected in evaluate_test_cases.items():
        result = evaluate_expression(expr)
        if result == expected:
            print(f"  Evaluate '{expr}' -> '{result}' (Correct)")
        else:
            print(f"  Evaluate '{expr}' -> '{result}' (Expected '{expected}', INCORRECT!)")

    # Test with non-string input for evaluate_expression
    print("\nNon-string input test for evaluate_expression:")
    if evaluate_expression(123) == "Error!": # type: ignore
        print(f"  `evaluate_expression(123)` (int) -> 'Error!' (Correct)")
    else:
        print(f"  `evaluate_expression(123)` (int) -> '{evaluate_expression(123)}' (INCORRECT!)") # type: ignore
