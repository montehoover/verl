import ast
import logging

# Configure basic logging if no handlers are configured
# This is a simple setup; in a larger application, logging might be configured centrally.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def safe_execute_operation(operation: str):
    """
    Evaluates a mathematical operation string safely.

    Args:
        operation: The mathematical operation string to be evaluated.

    Returns:
        The result of the evaluated operation.

    Raises:
        ValueError: If unsafe or invalid characters are detected,
                    the operation is not purely mathematical,
                    or if the evaluation logic encounters any issues.
    """
    logger.info(f"Attempting to execute operation: '{operation}'")

    try:
        # Parse the operation string into an Abstract Syntax Tree (AST)
        # 'eval' mode is used because we expect an expression.
        logger.debug(f"Parsing operation: '{operation}'")
        node = ast.parse(operation, mode='eval')
        logger.info(f"Successfully parsed operation: '{operation}'")
    except SyntaxError as e:
        logger.error(f"Syntax error parsing operation '{operation}': {e}")
        raise ValueError(f"Invalid syntax in operation: '{operation}'")
    except Exception as e:
        # Catch other potential parsing errors.
        logger.error(f"Generic error parsing operation '{operation}': {e}", exc_info=True)
        raise ValueError(f"Error parsing operation '{operation}': {e}")

    # --- AST Node Whitelisting ---
    # To prevent arbitrary code execution, we define a whitelist of allowed
    # AST node types and specific operators. Only nodes and operations
    # corresponding to simple arithmetic will be permitted.

    # Allowed AST node types. These are the fundamental building blocks
    # of the expression's structure that we permit.
    allowed_node_types = (
        ast.Expression,  # The root node for an expression.
        ast.Constant,    # For numeric literals (e.g., 5, 3.14).
        ast.BinOp,       # For binary operations (e.g., +, -, *, /).
        ast.UnaryOp,     # For unary operations (e.g., -5).
    )
    
    # Allowed binary operator types (used within ast.BinOp nodes).
    # These define the mathematical operations like +, -, *, etc.
    allowed_bin_op_types = (
        ast.Add,        # Addition (+)
        ast.Sub,        # Subtraction (-)
        ast.Mult,       # Multiplication (*)
        ast.Div,        # True division (/)
        ast.Pow,        # Exponentiation (**)
        ast.Mod,        # Modulo (%)
        ast.FloorDiv    # Floor division (//)
    )
    
    # Allowed unary operator types (used within ast.UnaryOp nodes).
    # These define operations like unary plus (+x) and unary minus (-x).
    allowed_unary_op_types = (
        ast.UAdd,       # Unary plus (+x)
        ast.USub        # Unary minus (-x)
    )

    # --- AST Traversal and Validation ---
    # Walk through the parsed AST and check each node against the whitelists.
    # This is crucial for security, ensuring no disallowed operations or
    # node types are present in the expression.
    logger.debug(f"Starting AST validation for operation: '{operation}'")
    for sub_node in ast.walk(node):
        node_type_name = type(sub_node).__name__
        logger.debug(f"Validating node: {sub_node} (Type: {node_type_name})")

        # Check 1: Ensure the node itself is of an allowed type (e.g., Expression, Constant, BinOp, UnaryOp).
        if not isinstance(sub_node, allowed_node_types):
            log_msg = (
                f"Unsafe operation: Node type '{node_type_name}' is not allowed. "
                f"Operation: '{operation}'"
            )
            logger.error(log_msg)
            raise ValueError(log_msg)

        # Check 2: For constants, ensure they are numeric (integers or floats).
        # This prevents evaluation of other constant types like strings or None, if they were to appear.
        if isinstance(sub_node, ast.Constant):
            constant_value_type_name = type(sub_node.value).__name__
            if not isinstance(sub_node.value, (int, float)):
                log_msg = (
                    f"Invalid constant: Only numeric constants (int, float) are allowed. "
                    f"Found type '{constant_value_type_name}' for value '{sub_node.value}'. "
                    f"Operation: '{operation}'"
                )
                logger.error(log_msg)
                raise ValueError(log_msg)
        # Check 3: For binary operations, ensure the specific operator (e.g., Add, Sub) is allowed.
        elif isinstance(sub_node, ast.BinOp):
            op_type_name = type(sub_node.op).__name__
            if not isinstance(sub_node.op, allowed_bin_op_types):
                log_msg = (
                    f"Unsafe binary operator: Operator type '{op_type_name}' is not allowed. "
                    f"Operation: '{operation}'"
                )
                logger.error(log_msg)
                raise ValueError(log_msg)
        # Check 4: For unary operations, ensure the specific operator (e.g., UAdd, USub) is allowed.
        elif isinstance(sub_node, ast.UnaryOp):
            op_type_name = type(sub_node.op).__name__
            if not isinstance(sub_node.op, allowed_unary_op_types):
                log_msg = (
                    f"Unsafe unary operator: Operator type '{op_type_name}' is not allowed. "
                    f"Operation: '{operation}'"
                )
                logger.error(log_msg)
                raise ValueError(log_msg)
    logger.info(f"AST validation successful for operation: '{operation}'")

    # --- Safe Evaluation ---
    # If all AST nodes and operations have been validated, proceed to compile and
    # evaluate the expression in a restricted environment.
    try:
        logger.debug(f"Compiling AST for operation: '{operation}'")
        # Compile the AST Expression node into a code object.
        # 'node' is an ast.Expression as returned by ast.parse(..., mode='eval').
        # The actual expression content is in node.body, but compile handles the Expression node directly.
        code_object = compile(node, filename='<string>', mode='eval')
        logger.debug(f"Successfully compiled AST for operation: '{operation}'")
        
        logger.debug(f"Evaluating compiled code for operation: '{operation}'")
        # Evaluate the compiled code.
        # Provide empty dictionaries for globals and locals to restrict the execution environment,
        # preventing access to built-in functions or other potentially unsafe operations.
        result = eval(code_object, {"__builtins__": {}}, {})
        logger.info(f"Successfully evaluated operation: '{operation}'. Result: {result}")
        return result
    except ZeroDivisionError:
        log_msg = f"Error during evaluation: Division by zero. Operation: '{operation}'"
        logger.error(log_msg)
        raise ValueError(log_msg)
    except Exception as e:
        # Catch any other errors during evaluation (e.g., overflow).
        log_msg = f"Error during evaluation of '{operation}': {e}"
        logger.error(log_msg, exc_info=True)
        raise ValueError(log_msg)
