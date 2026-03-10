import re
import ast
import operator as op

# Supported binary operators
_SUPPORTED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}

# Supported unary operators
_SUPPORTED_UNARY_OPERATORS = {
    ast.USub: op.neg,  # Unary minus
    ast.UAdd: op.pos,  # Unary plus
}

# Regular expression for allowed characters in the expression
_ALLOWED_CHARS_PATTERN = re.compile(r"^[0-9\s\.\+\-\*\/\(\)]+$")

def _eval_node(node):
    """
    Recursively evaluates an AST node for an arithmetic expression.
    Supports: numbers (int, float), +, -, *, /, unary -, unary +.
    Parentheses are handled by the AST structure.
    """
    if isinstance(node, ast.Constant):  # Python 3.8+
        if not isinstance(node.value, (int, float)):
            # This TypeError will be caught and re-raised as ValueError
            raise TypeError(f"Unsupported constant type: {type(node.value)}. Only int/float allowed.")
        return node.value
    elif isinstance(node, ast.Num):  # Python < 3.8
        if not isinstance(node.n, (int, float)):
            # This TypeError will be caught and re-raised as ValueError
            raise TypeError(f"Unsupported number type: {type(node.n)}. Only int/float allowed.")
        return node.n
    elif isinstance(node, ast.BinOp):
        operator_func = _SUPPORTED_OPERATORS.get(type(node.op))
        if operator_func is None:
            # This TypeError will be caught and re-raised as ValueError
            raise TypeError(f"Unsupported binary operator: {type(node.op)}")
        
        left_val = _eval_node(node.left)
        right_val = _eval_node(node.right)
        
        if isinstance(node.op, ast.Div) and right_val == 0:
            # This ZeroDivisionError will be caught and re-raised as ValueError
            raise ZeroDivisionError("division by zero")
            
        return operator_func(left_val, right_val)
    elif isinstance(node, ast.UnaryOp):
        operator_func = _SUPPORTED_UNARY_OPERATORS.get(type(node.op))
        if operator_func is None:
            # This TypeError will be caught and re-raised as ValueError
            raise TypeError(f"Unsupported unary operator: {type(node.op)}")
        
        operand_val = _eval_node(node.operand)
        return operator_func(operand_val)
    else:
        # This TypeError will be caught and re-raised as ValueError
        raise TypeError(
            f"Unsupported AST node type: {type(node).__name__}. "
            "Only numbers, basic arithmetic operators (+, -, *, /), "
            "unary minus/plus, and parentheses are allowed."
        )

def parse_and_calculate(expression: str) -> float:
    """
    Parses and calculates a mathematical expression string.

    The function supports basic arithmetic operations (+, -, *, /),
    numbers (integers and floats), parentheses for grouping, and
    unary plus/minus.

    Args:
        expression: The mathematical expression string to evaluate.
                    (e.g., "4 + 3", "-(5 - 2) * 3 / 2.5").

    Returns:
        The result of the calculation as a float.

    Raises:
        ValueError: If the expression string is empty, contains unsupported
                    characters, is syntactically incorrect, attempts an unsafe
                    operation (like division by zero), or includes unsupported
                    mathematical constructs (e.g., variables, functions).
    """
    if not isinstance(expression, str):
        # Technically, type hints should catch this, but good for runtime safety.
        raise ValueError("Expression must be a string.")
    
    stripped_expression = expression.strip()
    if not stripped_expression:
        raise ValueError("Expression cannot be empty.")

    if not _ALLOWED_CHARS_PATTERN.match(stripped_expression):
        raise ValueError("Expression contains unsupported characters.")

    try:
        # ast.parse with mode='eval' expects a single expression.
        # The result is an ast.Expression node, whose 'body' attribute
        # contains the actual root node of the expression.
        ast_tree = ast.parse(stripped_expression, mode='eval')
        result = _eval_node(ast_tree.body)
        
        # _eval_node should always return a number (int or float) or raise.
        # This check is a safeguard.
        if not isinstance(result, (int, float)):
            # This case should ideally be caught by _eval_node's type checks.
            raise ValueError(f"Calculation resulted in a non-numeric type: {type(result).__name__}")
            
        return float(result)
    except (SyntaxError, TypeError, ZeroDivisionError) as e:
        # Catch specific errors from parsing or evaluation and re-raise as ValueError.
        # SyntaxError: Malformed expression (e.g., "1 +", "(2 * (3").
        # TypeError: Unsupported operations/types within _eval_node.
        # ZeroDivisionError: Division by zero.
        raise ValueError(f"Invalid or unsafe expression: {e}")
    except Exception as e:
        # Catch any other unexpected errors during parsing or evaluation.
        # This could include RecursionError for deeply nested expressions if not handled.
        # Re-raise as ValueError for consistent error reporting.
        raise ValueError(f"Unexpected error processing expression: {e}")
