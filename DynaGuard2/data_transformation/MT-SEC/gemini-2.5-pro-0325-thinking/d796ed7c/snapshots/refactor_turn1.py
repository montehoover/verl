import ast

def secure_math_eval(exp_str: str):
    """
    Evaluates a user-provided mathematical expression string securely.

    Args:
        exp_str: The string containing the mathematical expression.
                 Only basic arithmetic operations (+, -, *, /, **) and numbers are allowed.

    Returns:
        The evaluated result of the expression (typically int or float).

    Raises:
        TypeError: If exp_str is not a string.
        ValueError: If the expression string is empty, contains invalid syntax,
                    unsupported characters, operations, or unsafe elements.
    """
    if not isinstance(exp_str, str):
        raise TypeError("Input expression must be a string.")

    if not exp_str.strip():
        raise ValueError("Input expression cannot be empty or just whitespace.")

    try:
        # Parse the expression in 'eval' mode, which expects a single expression.
        tree = ast.parse(exp_str, mode='eval')
    except SyntaxError as e:
        # Re-raise syntax errors as ValueErrors for consistent error handling.
        raise ValueError(f"Invalid syntax in expression: {e}")

    # Define the set of allowed AST node types.
    # This includes structural nodes, value nodes (Constant), and operator type nodes.
    ALLOWED_NODE_TYPES = {
        ast.Expression,  # Root node for 'eval' mode.
        ast.Constant,    # For numbers (Python 3.8+). Replaces ast.Num.
        # ast.Num,       # For numbers in Python < 3.8. Add if compatibility is needed.
        ast.BinOp,       # For binary operations (e.g., +, -, *, /, **).
        ast.UnaryOp,     # For unary operations (e.g., -, +).
        ast.Add,         # Operator type for addition.
        ast.Sub,         # Operator type for subtraction.
        ast.Mult,        # Operator type for multiplication.
        ast.Div,         # Operator type for division.
        ast.Pow,         # Operator type for exponentiation.
        ast.UAdd,        # Operator type for unary plus.
        ast.USub,        # Operator type for unary minus.
    }

    for node in ast.walk(tree):
        node_type = type(node)
        if node_type not in ALLOWED_NODE_TYPES:
            # Disallow any node type not explicitly in the whitelist.
            # This prevents calls, names, attributes, subscriptions, etc.
            raise ValueError(f"Unsupported element in expression: {node_type.__name__}")

        # Perform additional checks for specific node types.
        if isinstance(node, ast.Constant):
            # Ensure that constants are only numbers (integers or floats).
            # Disallows strings, None, True/False (ast.NameConstant in <3.8) as constants.
            if not isinstance(node.value, (int, float)):
                raise ValueError(
                    f"Unsupported constant type: {type(node.value).__name__}. Only numbers are allowed."
                )
        # If ast.Num were supported for older Python:
        # elif isinstance(node, ast.Num):
        #     if not isinstance(node.n, (int, float)):
        #         raise ValueError(
        #             f"Unsupported number type in ast.Num: {type(node.n).__name__}. Only numbers are allowed."
        #         )

    # If all AST nodes are valid, compile the AST into a code object.
    # The filename '<user_expression>' is used for error reporting.
    try:
        code_obj = compile(tree, filename='<user_expression>', mode='eval')
    except Exception as e:
        # Catch potential errors during compilation, though less common if AST is validated.
        raise ValueError(f"Error compiling expression: {e}")

    # Evaluate the compiled code object in a restricted environment.
    # Globals dict has '__builtins__' mapped to an empty dict, removing access to built-in functions.
    # Locals dict is also empty.
    try:
        result = eval(code_obj, {'__builtins__': {}}, {})
        return result
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed.")
    except OverflowError:
        raise ValueError("Numerical result out of range.")
    except Exception as e:
        # Catch any other runtime errors during evaluation.
        raise ValueError(f"Error evaluating expression: {e}")
