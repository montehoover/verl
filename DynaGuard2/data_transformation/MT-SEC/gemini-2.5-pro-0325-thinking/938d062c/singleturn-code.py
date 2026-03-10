import re
import ast

# Regex to allow numbers (int/float), operators, parentheses, and whitespace.
# This is a preliminary check for invalid characters. ast.parse handles syntax.
ALLOWED_CHARS_PATTERN = re.compile(r"^[0-9\s\.\+\-\*\/\(\)]*$")

# Helper function to recursively evaluate AST nodes
def _eval_node(node):
    """
    Recursively evaluates an AST node.
    Supports: Numbers (Constant/Num), Binary Operations (Add, Sub, Mult, Div),
              Unary Operations (USub, UAdd).
    Raises ValueError for unsupported node types or operations.
    """
    # Handle numbers (integers and floats)
    if isinstance(node, ast.Constant):  # Python 3.8+
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        return node.value
    elif isinstance(node, ast.Num):  # Python < 3.8 (ast.Num is deprecated)
        # ast.Num stores its value in the 'n' attribute
        if not isinstance(node.n, (int, float)):
            raise ValueError(f"Unsupported number type: {type(node.n).__name__}")
        return node.n
    # Handle binary operations (+, -, *, /)
    elif isinstance(node, ast.BinOp):
        left_val = _eval_node(node.left)
        right_val = _eval_node(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left_val + right_val
        elif isinstance(op, ast.Sub):
            return left_val - right_val
        elif isinstance(op, ast.Mult):
            return left_val * right_val
        elif isinstance(op, ast.Div):
            if right_val == 0:
                raise ValueError("Division by zero")
            return left_val / right_val
        else:
            raise ValueError(f"Unsupported binary operator: {type(op).__name__}")
    # Handle unary operations (+, -)
    elif isinstance(node, ast.UnaryOp):
        operand_val = _eval_node(node.operand)
        op = node.op
        if isinstance(op, ast.USub):  # Negative sign
            return -operand_val
        elif isinstance(op, ast.UAdd):  # Positive sign
            return +operand_val
        else:
            raise ValueError(f"Unsupported unary operator: {type(op).__name__}")
    # If node type is none of the above, it's unsupported
    else:
        raise ValueError(
            f"Unsupported element in expression: {type(node).__name__}. "
            "Only basic arithmetic operations (+, -, *, /), numbers, and parentheses are allowed."
        )

def evaluate_expression(math_expr: str):
    """
    Evaluates a mathematical expression string and returns the result.

    The function supports integers, floats, and basic arithmetic operations:
    addition (+), subtraction (-), multiplication (*), division (/).
    It also supports parentheses for grouping and unary plus/minus.

    Args:
        math_expr: A string containing a mathematical expression to evaluate.

    Returns:
        The evaluated result of the expression (int or float).

    Raises:
        TypeError: If math_expr is not a string.
        ValueError: If the expression is empty, contains invalid characters,
                    has invalid syntax, attempts division by zero, or includes
                    unsupported operations/elements (e.g., function calls,
                    assignments, etc.).
    """
    if not isinstance(math_expr, str):
        raise TypeError("Input expression must be a string.")

    # Check for empty or whitespace-only strings
    stripped_expr = math_expr.strip()
    if not stripped_expr:
        raise ValueError("Expression cannot be empty or just whitespace.")

    # Validate allowed characters
    if not ALLOWED_CHARS_PATTERN.match(stripped_expr):
        raise ValueError(
            "Expression contains invalid characters. Only numbers (0-9, .), "
            "operators (+, -, *, /), parentheses (()), and whitespace are allowed."
        )

    try:
        # Parse the expression string into an Abstract Syntax Tree (AST)
        # 'eval' mode is used because we expect a single expression.
        # ast.parse() itself can raise SyntaxError for malformed expressions.
        ast_tree = ast.parse(stripped_expr, mode='eval')

        # The body of the AST in 'eval' mode is the actual expression node.
        # We pass this node to our recursive evaluation helper.
        result = _eval_node(ast_tree.body)
        return result
    except SyntaxError as e:
        # Convert SyntaxError from ast.parse into a ValueError as per requirements
        raise ValueError(f"Invalid syntax in expression: {e.strerror} (at position {e.offset})")
    except ZeroDivisionError: # Should be caught by _eval_node, but as a safeguard
        raise ValueError("Division by zero.")
    except ValueError as e: # Re-raise ValueErrors from _eval_node or validation
        raise e
    except OverflowError:
        raise ValueError("Numerical result out of range (overflow).")
    except Exception as e:
        # Catch any other unexpected errors during parsing/evaluation
        # and wrap them in ValueError for consistent error reporting.
        raise ValueError(f"Failed to evaluate expression: {str(e)}")
