import re
import ast

# Whitelist of allowed AST node types.
# This is crucial for security to prevent arbitrary code execution.
# It ensures that only nodes corresponding to basic arithmetic are processed.
ALLOWED_NODE_TYPES = (
    ast.Expression,  # Represents an entire expression.
    ast.Constant,    # For literal values like numbers (Python 3.8+).
                     # In Python <3.8, ast.Num was used for numbers.
    ast.Num,         # For numbers (used in Python < 3.8, ast.Constant preferred in 3.8+).
    ast.BinOp,       # For binary operations (e.g., a + b).
    ast.UnaryOp,     # For unary operations (e.g., -a).
    ast.Add,         # The + operator.
    ast.Sub,         # The - operator.
    ast.Mult,        # The * operator.
    ast.Div,         # The / operator.
    ast.Pow,         # The ** operator (exponentiation).
    ast.UAdd,        # Unary + operator (e.g., +5).
    ast.USub,        # Unary - operator (e.g., -5).
)

def compute_expression(expr: str):
    """
    Evaluates a mathematical expression string and returns the result.

    The function parses the input string into an Abstract Syntax Tree (AST),
    validates that all nodes in the AST correspond to allowed mathematical
    operations or numeric constants, and then compiles and evaluates the AST.

    Args:
        expr: str, a string containing a mathematical expression to evaluate.
              Example: "2 + 3 * (4 - 1)"

    Returns:
        The evaluated result of the expression (int or float).

    Raises:
        TypeError: If the input `expr` is not a string.
        ValueError: If the expression is empty, contains invalid syntax,
                    includes unsupported operators or characters (e.g., function calls,
                    variable names), involves non-numeric constants, or if an error
                    occurs during evaluation (e.g., division by zero, overflow).
    """
    if not isinstance(expr, str):
        raise TypeError("Expression must be a string.")
    
    if not expr.strip():
        raise ValueError("Expression cannot be empty or consist only of whitespace.")

    try:
        # Parse the expression string into an Abstract Syntax Tree (AST).
        # 'eval' mode is used because we expect a single expression that returns a value.
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        # If parsing fails (e.g., mismatched parentheses, invalid characters),
        # it's an invalid expression.
        raise ValueError(f"Invalid syntax in expression: {e}") from e

    # Validate all nodes in the AST to ensure they are on the whitelist.
    # ast.walk traverses the AST, yielding all nodes.
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODE_TYPES):
            error_message = (
                f"Unsupported element or operation in expression: {type(node).__name__}. "
                "Only basic arithmetic operations (+, -, *, /, **), "
                "and numbers are allowed."
            )
            raise ValueError(error_message)
        
        # Additional check for ast.Constant: ensure it holds a number.
        # ast.Constant can also represent strings, None, True, False in a general AST.
        if isinstance(node, ast.Constant):
            # This inner 'if' is a guard clause for the value of an ast.Constant node.
            if not isinstance(node.value, (int, float)):
                error_message = (
                    f"Unsupported constant type: {type(node.value).__name__}. "
                    "Only numeric constants (integers or floats) are allowed."
                )
                raise ValueError(error_message)
        # For Python versions older than 3.8, ast.Num was used for numeric constants.
        # Using elif as ast.Constant and ast.Num are distinct types.
        elif isinstance(node, ast.Num): 
            # This inner 'if' is a guard clause for the value of an ast.Num node.
            if not isinstance(node.n, (int, float)):
                error_message = (
                    f"Unsupported number type in ast.Num node: {type(node.n).__name__}. "
                    "Only numeric constants are allowed."
                )
                raise ValueError(error_message)

    # If all nodes are valid, compile the AST into a code object.
    # The filename '<expression>' is used for more informative error messages.
    try:
        code = compile(tree, filename='<expression>', mode='eval')
    except Exception as e:
        # Catch potential errors during compilation, though less common for valid ASTs.
        raise ValueError(f"Error compiling expression: {e}") from e

    # Evaluate the compiled code object.
    # Provide an empty dictionary for globals and a restricted dictionary for locals
    # (specifically, __builtins__ is emptied) to create a very restricted execution
    # environment. This prevents access to arbitrary built-in functions or variables.
    try:
        result = eval(code, {"__builtins__": {}}, {})
        return result
    except ZeroDivisionError:
        raise ValueError("Evaluation error: Division by zero is not allowed.")
    except OverflowError:
        raise ValueError("Evaluation error: Numerical result is out of range.")
    except Exception as e:
        # Catch any other unexpected errors during evaluation.
        raise ValueError(f"Error evaluating expression: {e}") from e
