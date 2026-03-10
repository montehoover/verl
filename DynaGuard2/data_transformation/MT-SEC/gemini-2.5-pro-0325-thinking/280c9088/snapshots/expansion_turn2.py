import ast

def validate_expression(expression: str) -> bool:
    """
    Validates a mathematical expression string to ensure it only contains
    basic arithmetic operations (+, -, *, /), numbers, and parentheses,
    and is safe for evaluation.
    """
    if not isinstance(expression, str):
        # Input must be a string.
        return False
    if not expression.strip():
        # Empty or whitespace-only strings are not valid expressions.
        return False

    try:
        # Parse the expression in 'eval' mode, suitable for single expressions.
        tree = ast.parse(expression, mode='eval')
    except (SyntaxError, TypeError, ValueError):
        # SyntaxError: Malformed expression (e.g., "1 +", "())").
        # TypeError: Input is not a string (e.g., None), though checked above.
        # ValueError: e.g., "Expression too large", "malformed string node".
        return False

    # Whitelist of allowed AST node types.
    # ast.Num is for numbers in Python < 3.8.
    # ast.Constant is for numbers, strings, None, bools in Python 3.8+.
    # We will specifically check ast.Constant's value type.
    allowed_node_types = (
        ast.Expression,  # The root node for an expression.
        ast.Constant,    # Represents literal values like numbers, strings.
        ast.Num,         # Represents numbers (used in older Python versions).
        ast.BinOp,       # Represents binary operations (e.g., a + b).
        ast.UnaryOp,     # Represents unary operations (e.g., -a).
    )

    # Whitelist of allowed operator types within BinOp and UnaryOp nodes.
    allowed_operator_types = (
        ast.Add,         # For +
        ast.Sub,         # For -
        ast.Mult,        # For *
        ast.Div,         # For /
        ast.UAdd,        # For unary + (e.g., +5)
        ast.USub,        # For unary - (e.g., -5)
    )

    for node in ast.walk(tree):
        # Check if the node type is in our allowed list.
        if not isinstance(node, allowed_node_types):
            # If not, this could be ast.Name, ast.Call, ast.Attribute, etc.,
            # which are disallowed for safety.
            return False

        # Additional checks for specific node types:
        if isinstance(node, ast.Constant):
            # If it's an ast.Constant, ensure its value is a number (int or float).
            # This prevents evaluation of strings, booleans, or None if they parse as Constants.
            if not isinstance(node.value, (int, float)):
                return False
        elif isinstance(node, ast.Num): # Relevant for Python < 3.8
            # If it's an ast.Num, ensure its value (node.n) is a number.
            # This is mostly a sanity check as ast.Num should always hold a number.
            if not isinstance(node.n, (int, float)):
                return False
        elif isinstance(node, (ast.BinOp, ast.UnaryOp)):
            # If it's a binary or unary operation, ensure the operator itself is allowed.
            # This disallows operators like ast.Pow (**), ast.Mod (%), etc.
            if not isinstance(node.op, allowed_operator_types):
                return False
        # No specific checks needed for ast.Expression as it's a container.
        # ast.Add, ast.Sub, etc., are types of the 'op' attribute, not separate nodes.

    # If all nodes and operators are allowed, the expression is considered valid and safe.
    return True


def evaluate_safe_expression(expression: str):
    """
    Evaluates a pre-validated mathematical expression string.

    Args:
        expression: A string containing a mathematical expression that has already
                    been validated by `validate_expression`.

    Returns:
        The numerical result of the evaluation if successful.
        An error message string if any computation error occurs (e.g., division by zero).
    """
    try:
        # Since the expression is pre-validated, we can use eval().
        # The validation ensures no unsafe code (like function calls or variable names)
        # is present, only basic arithmetic.
        result = eval(expression)
        if isinstance(result, (int, float)):
            return result
        else:
            # This case should ideally not be reached if validation is robust
            # and only allows operations resulting in numbers.
            return "Error: Evaluation resulted in a non-numeric type."
    except ZeroDivisionError:
        return "Error: Division by zero."
    except OverflowError:
        return "Error: Numerical result out of range."
    except Exception as e:
        # Catch any other unexpected errors during evaluation.
        # This is a fallback, as validate_expression should prevent most issues.
        return f"Error: An unexpected error occurred during computation - {type(e).__name__}."
