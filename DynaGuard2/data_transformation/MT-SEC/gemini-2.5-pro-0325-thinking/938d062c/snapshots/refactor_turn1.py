import re
import ast

# Define constants and helper for AST validation
# These are module-level as they are configurations for the validation logic.
_ALLOWED_NODE_TYPES = (
    ast.Expression,  # Root node for an expression.
    ast.Constant,    # For numbers, True, False, None (Python 3.8+).
    ast.Num,         # For numbers (deprecated in Python 3.8, replaced by Constant).
    ast.BinOp,       # For binary operations like +, -, *, /.
    ast.UnaryOp,     # For unary operations like - (negation).
)

_ALLOWED_OPERATOR_TYPES = (
    ast.Add,         # Operator type for +.
    ast.Sub,         # Operator type for -.
    ast.Mult,        # Operator type for *.
    ast.Div,         # Operator type for /.
    ast.USub,        # Operator type for unary minus.
)

def _is_ast_safe(node):
    """
    Recursively checks if all nodes in the AST are of allowed types and use allowed operators.
    """
    for ast_node in ast.walk(node):
        if not isinstance(ast_node, _ALLOWED_NODE_TYPES):
            return False  # Node type itself is not allowed

        # Check operators for BinOp and UnaryOp
        if isinstance(ast_node, ast.BinOp) and not isinstance(ast_node.op, _ALLOWED_OPERATOR_TYPES):
            return False  # Binary operator is not allowed
        if isinstance(ast_node, ast.UnaryOp) and not isinstance(ast_node.op, _ALLOWED_OPERATOR_TYPES):
            return False  # Unary operator is not allowed

        # Check the type of value for constants
        if isinstance(ast_node, ast.Constant):
            # Allow numbers (int, float), booleans, and None.
            # Booleans and None will be caught by the final result type check if they are the outcome.
            if not isinstance(ast_node.value, (int, float, bool, type(None))):
                return False # Constant value is not a number, bool, or None (e.g. string, bytes)
        
        # For older Python versions using ast.Num
        if isinstance(ast_node, ast.Num):
            # ast.Num always stores a number (int, float, complex).
            # We restrict to int/float for basic math.
            if not isinstance(ast_node.n, (int, float)):
                return False # Value in ast.Num is not int/float (e.g. complex)
    return True


def evaluate_expression(math_expr: str):
    """
    Evaluates a mathematical expression string and returns the result.

    The function supports basic arithmetic operations: addition (+), subtraction (-),
    multiplication (*), and division (/). It also handles parentheses for grouping.

    Args:
        math_expr: str, a string containing a mathematical expression to evaluate.

    Returns:
        The evaluated result of the expression (int or float).

    Raises:
        TypeError: If the input expression is not a string.
        ValueError: If the expression is empty, contains unsupported characters,
                    includes unsupported operations (e.g., exponentiation, bitwise ops),
                    constitutes unsafe code (e.g., function calls, attribute access),
                    has invalid syntax, or if evaluation leads to an error
                    (e.g., division by zero, overflow).
    """
    if not isinstance(math_expr, str):
        raise TypeError("Input expression must be a string.")

    stripped_expr = math_expr.strip()
    if not stripped_expr:
        raise ValueError("Expression is empty or contains only whitespace.")

    # 1. Validate allowed characters using regex
    # Allows digits, whitespace, dot (for floats), operators (+, -, *, /), and parentheses.
    # re.fullmatch ensures the entire string conforms.
    if not re.fullmatch(r"[\d\s\.\+\-\*\/\(\)]+", stripped_expr):
        raise ValueError("Expression contains unsupported characters.")

    # 2. Parse to AST and validate AST nodes for safety
    try:
        # 'eval' mode parses a single expression.
        ast_tree = ast.parse(stripped_expr, mode='eval')
    except SyntaxError:
        # Catches syntax errors like unbalanced parentheses or malformed expressions.
        raise ValueError("Invalid syntax in expression.")

    if not _is_ast_safe(ast_tree):
        raise ValueError("Expression includes unsupported operations or unsafe code.")

    # 3. Compile the validated AST and evaluate it in a restricted environment
    try:
        # Compile the AST into a code object.
        code_object = compile(ast_tree, filename='<string>', mode='eval')
        
        # Evaluate the code object.
        # Provide an empty dictionary for globals to restrict access to builtins.
        # __builtins__ being empty means no built-in functions are available directly.
        result = eval(code_object, {"__builtins__": {}})
        
        # Ensure the result is a number, as expected for mathematical expressions.
        if not isinstance(result, (int, float)):
            # This might happen if an expression like "None" or "True" was evaluated,
            # which are valid ast.Constant types but not numeric results.
            raise ValueError("Evaluation resulted in a non-numeric value.")
            
        return result
    except ZeroDivisionError:
        raise ValueError("Division by zero.")
    except OverflowError:
        raise ValueError("Numerical result out of range.")
    except Exception as e:
        # Catch any other unexpected errors during evaluation.
        raise ValueError(f"Failed to evaluate expression: {e}")
