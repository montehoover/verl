"""
Module for securely evaluating mathematical expressions.
"""
import ast
import logging

# Configure a logger for this module.
# It's generally better to configure logging at the application entry point,
# but if required within the function, we'll handle it there.
# For this example, we'll set up a basic logger configuration.
# If the function is part of a larger application, this top-level configuration
# might be preferred or handled by the application's logging setup.

# Create a logger instance for this specific function/module
secure_eval_logger = logging.getLogger('secure_math_eval')
secure_eval_logger.setLevel(logging.INFO) # Default level

# Prevent adding multiple handlers if this module is reloaded or function called multiple times
if not secure_eval_logger.handlers:
    # Create a file handler to log messages to a file
    # In a real application, the log file path and configuration would be more robust.
    fh = logging.FileHandler('secure_math_eval.log')
    fh.setLevel(logging.INFO)

    # Create a console handler for outputting to console (optional)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.ERROR)

    # Create a formatter and set it for the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)

    # Add the handlers to the logger
    secure_eval_logger.addHandler(fh)
    # secure_eval_logger.addHandler(ch)


def secure_math_eval(exp_str: str):
    """
    Evaluates a user-provided mathematical expression string securely.

    This function parses the input string, validates its Abstract Syntax Tree (AST)
    to allow only basic arithmetic operations and numeric constants, and then
    evaluates the expression in a restricted environment.

    Args:
        exp_str: The string containing the mathematical expression.
                 Only basic arithmetic operations (+, -, *, /, **) and numbers are allowed.

    Returns:
        The evaluated result of the expression (typically int or float).

    Raises:
        TypeError: If exp_str is not a string.
        ValueError: If the expression string is empty, contains invalid syntax,
                    unsupported characters/operations, or unsafe elements.
    """
    secure_eval_logger.info(f"Attempting to evaluate expression: '{exp_str}'")

    # Validate input type and content
    if not isinstance(exp_str, str):
        secure_eval_logger.error(f"TypeError: Input expression must be a string. Received: {type(exp_str)}")
        raise TypeError("Input expression must be a string.")

    if not exp_str.strip():
        secure_eval_logger.error("ValueError: Input expression cannot be empty or just whitespace.")
        raise ValueError("Input expression cannot be empty or just whitespace.")

    # Attempt to parse the expression string into an AST
    try:
        # 'eval' mode is used because we expect a single expression.
        tree = ast.parse(exp_str, mode='eval')
    except SyntaxError as e:
        # Re-raise syntax errors as ValueErrors for consistent API.
        secure_eval_logger.error(f"SyntaxError during parsing of '{exp_str}': {e}")
        raise ValueError(f"Invalid syntax in expression: {e}")

    # Define the whitelist of allowed AST node types.
    # This is crucial for security, as it restricts what operations and
    # structures are permitted in the expression.
    ALLOWED_NODE_TYPES = {
        ast.Expression,  # The root node of an expression parsed in 'eval' mode.
        ast.Constant,    # Represents literal values like numbers (Python 3.8+).
                         # For Python < 3.8, ast.Num would be used for numbers.
        # ast.Num,       # Uncomment if Python < 3.8 compatibility for numbers is needed.
        ast.BinOp,       # Represents binary operations (e.g., a + b, a * b).
        ast.UnaryOp,     # Represents unary operations (e.g., -a).
        ast.Add,         # Specific operator type for addition.
        ast.Sub,         # Specific operator type for subtraction.
        ast.Mult,        # Specific operator type for multiplication.
        ast.Div,         # Specific operator type for division.
        ast.Pow,         # Specific operator type for exponentiation (**).
        ast.UAdd,        # Specific operator type for unary plus (+a).
        ast.USub,        # Specific operator type for unary minus (-a).
    }

    # Traverse the AST to validate all nodes
    for node in ast.walk(tree):
        node_type = type(node)
        if node_type not in ALLOWED_NODE_TYPES:
            # If any node is not in our whitelist, reject the expression.
            # This prevents use of functions, variable names, attribute access, etc.
            err_msg = f"Unsupported element type in expression: {node_type.__name__}"
            secure_eval_logger.error(f"ValueError for expression '{exp_str}': {err_msg}")
            raise ValueError(err_msg)

        # Additional checks for specific node types
        if isinstance(node, ast.Constant):
            # Ensure that constants are only numeric (integers or floats).
            # This disallows string literals, None, True/False as constants within the expression.
            if not isinstance(node.value, (int, float)):
                err_msg = (
                    f"Unsupported constant type: {type(node.value).__name__}. "
                    "Only numeric constants are allowed."
                )
                secure_eval_logger.error(f"ValueError for expression '{exp_str}': {err_msg}")
                raise ValueError(err_msg)
        # Example for older Python versions using ast.Num:
        # elif isinstance(node, ast.Num):
        #     # Ensure that ast.Num nodes (used in Python < 3.8 for numbers)
        #     # also contain only integers or floats.
        #     if not isinstance(node.n, (int, float)):
        #         raise ValueError(
        #             f"Unsupported number type in ast.Num: {type(node.n).__name__}. "
        #             "Only numeric constants are allowed."
        #         )

    # If AST validation passes, compile the AST into a code object.
    # The filename '<user_expression>' is used for better error messages if compilation fails.
    try:
        code_obj = compile(tree, filename='<user_expression>', mode='eval')
    except Exception as e:
        # Catch potential errors during compilation, though less common if AST is well-formed.
        secure_eval_logger.error(f"Error compiling expression '{exp_str}': {e}")
        raise ValueError(f"Error compiling expression: {e}")

    # Evaluate the compiled code object in a restricted environment.
    # The globals dictionary has '__builtins__' mapped to an empty dictionary,
    # effectively removing access to all built-in functions and names.
    # The locals dictionary is also empty.
    try:
        result = eval(code_obj, {'__builtins__': {}}, {})
        secure_eval_logger.info(f"Successfully evaluated expression '{exp_str}'. Result: {result}")
        return result
    except ZeroDivisionError:
        # Handle specific, common arithmetic errors.
        err_msg = "Division by zero is not allowed."
        secure_eval_logger.error(f"ZeroDivisionError for expression '{exp_str}': {err_msg}")
        raise ValueError(err_msg)
    except OverflowError:
        err_msg = "Numerical result out of range (overflow)."
        secure_eval_logger.error(f"OverflowError for expression '{exp_str}': {err_msg}")
        raise ValueError(err_msg)
    except Exception as e:
        # Catch any other unexpected errors during evaluation.
        secure_eval_logger.error(f"Error evaluating expression '{exp_str}': {e}")
        raise ValueError(f"Error evaluating expression: {e}")
