import ast

def secure_eval_expression(exp: str):
    """Evaluates a mathematical expression string securely.

    This function parses a user-provided string, checks for allowed
    mathematical operations using Abstract Syntax Trees (AST), and then
    evaluates the expression if it's deemed safe.

    Args:
        exp: A user-provided string containing the mathematical operation.
             Example: "2 + 3 * 4"

    Returns:
        The evaluated result of the provided mathematical expression.
        For "2 + 3 * 4", this would be 14.

    Raises:
        ValueError: If the expression string is empty, contains invalid
                    syntax, or uses operations/characters not explicitly
                    allowed (e.g., function calls, attribute access).
                    Also raised for evaluation errors like division by zero.
    """
    # Ensure the expression is not empty or just whitespace before parsing
    if not exp.strip():
        raise ValueError("Expression cannot be empty.")

    try:
        # Parse the expression string into an AST node.
        # 'eval' mode is used because we expect a single expression.
        node = ast.parse(exp, mode='eval')
    except SyntaxError:
        raise ValueError("Invalid syntax in expression.")

    # Define a whitelist of AST node types that are permitted.
    # This is crucial for security, as it prevents execution of arbitrary code.
    allowed_nodes = {
        ast.Expression,  # The root node of an expression.
        ast.Constant,    # For numbers, strings (Python 3.8+). Includes ast.Num.
        ast.Num,         # For numbers (deprecated in 3.8, use Constant).
        ast.BinOp,       # For binary operations like +, -, *, /.
        ast.UnaryOp,     # For unary operations like - (negation).
        ast.Add,         # Specific operator type: addition.
        ast.Sub,         # Specific operator type: subtraction.
        ast.Mult,        # Specific operator type: multiplication.
        ast.Div,         # Specific operator type: division.
        ast.Pow,         # Specific operator type: power.
        ast.Mod,         # Specific operator type: modulo.
        ast.FloorDiv,    # Specific operator type: floor division.
        ast.USub,        # Specific operator type: unary subtraction.
        ast.UAdd,        # Specific operator type: unary addition.
    }
    # For Python < 3.8, ast.NameConstant (True, False, None) might be needed
    # if those constants were to be allowed. Currently, they are not.
    # ast.Str for string literals (if allowed, currently not for math).

    # Traverse all nodes in the AST.
    # If any node is not in our whitelist, raise a ValueError.
    for sub_node in ast.walk(node):
        if type(sub_node) not in allowed_nodes:
            disallowed_type = type(sub_node).__name__
            raise ValueError(
                f"Unsafe operation or character: {disallowed_type}"
            )

    # Compile the AST node into a code object.
    # This step also performs some validation.
    # The filename '<string>' is a convention for code compiled from a string.
    try:
        code = compile(node, filename='<string>', mode='eval')
    except (TypeError, ValueError) as e:
        # Catch errors during compilation, e.g., if AST is malformed
        # in a way not caught by ast.parse or the node walk.
        raise ValueError(f"Invalid expression structure: {e}")

    # Evaluate the compiled code object.
    # A restricted environment is provided to eval:
    #   - globals: {"__builtins__": {}} disallows access to built-in functions.
    #   - locals: {} provides an empty local scope.
    # This is a critical security measure.
    try:
        return eval(code, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        raise ValueError("Error during evaluation: Division by zero.")
    except Exception as e:
        # Catch any other unexpected errors during evaluation.
        raise ValueError(f"Error during evaluation: {e}")
