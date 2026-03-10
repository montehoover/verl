import ast
import logging

# Set up a logger for this module
logger = logging.getLogger(__name__)

# Whitelist of allowed AST node types that can appear in the script.
# ast.Constant covers numbers, strings, True, False, None (Python 3.8+).
_ALLOWED_NODE_TYPES = (
    ast.Expression,  # The root of an evaluatable expression.
    ast.Constant,    # e.g., 123, "hello", True, None.
    ast.Name,        # Only for 'True', 'False', 'None' if not parsed as Constant.
    ast.UnaryOp,     # e.g., -x.
    ast.BinOp,       # e.g., x + y.
    ast.Compare,     # e.g., x == y, x < y.
    ast.BoolOp,      # e.g., x and y.
)

# Whitelists for specific operator types within compound AST nodes.
_ALLOWED_UNARY_OP_TYPES = (ast.UAdd, ast.USub)  # +x, -x
_ALLOWED_BIN_OP_TYPES = (
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow
) # +, -, *, /, //, %, **
_ALLOWED_COMPARE_OP_TYPES = (
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE
) # ==, !=, <, <=, >, >=
_ALLOWED_BOOL_OP_TYPES = (ast.And, ast.Or) # and, or


def _is_script_safe(tree: ast.AST) -> bool:
    """
    Validates if the AST tree contains only allowed operations.

    Args:
        tree: The AST tree to validate.

    Returns:
        True if the script is safe, False otherwise.
    """
    for node in ast.walk(tree):
        # Check if the node type itself is allowed.
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            return False

        # Specific checks for node types that have operators or names.
        if isinstance(node, ast.Name):
            # Only allow 'True', 'False', 'None' as names.
            if node.id not in {'True', 'False', 'None'}:
                return False
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARY_OP_TYPES):
                return False
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BIN_OP_TYPES):
                return False
        elif isinstance(node, ast.Compare):
            for op_type in node.ops: # Compare nodes can have multiple operators
                if not isinstance(op_type, _ALLOWED_COMPARE_OP_TYPES):
                    return False
        elif isinstance(node, ast.BoolOp):
            if not isinstance(node.op, _ALLOWED_BOOL_OP_TYPES):
                return False
        
        # Nodes like ast.Call, ast.Attribute, ast.Subscript, ast.List, ast.Dict,
        # ast.Lambda, ast.comprehensions, ast.FunctionDef, ast.ClassDef, ast.Import,
        # ast.Assign, etc., are blocked by the initial isinstance(node, _ALLOWED_NODE_TYPES) check.
    return True


def _execute_safe_script(tree: ast.AST) -> str:
    """
    Executes the validated AST tree in a restricted environment.

    Args:
        tree: The AST tree to execute.

    Returns:
        The result of the execution as a string.

    Raises:
        Exception: If any error occurs during compilation or evaluation.
    """
    # The environment for eval is highly restricted:
    #   globals: '__builtins__' is empty, blocking access to builtin functions.
    #   locals: empty, no predefined local variables.
    # Arithmetic and basic operations are intrinsic and don't rely on __builtins__.
    # ast.Constant handles True/False/None, so they are available.
    compiled_code = compile(tree, filename='<user_script>', mode='eval')
    result = eval(compiled_code, {'__builtins__': {}}, {})
    return str(result)


def process_user_code(code_script: str) -> str:
    """
    Securely parses and executes a Python script string containing basic arithmetic or string operations.

    Args:
        code_script: A Python script submitted by the user, intended for simple
                     arithmetic or string manipulations.

    Returns:
        The evaluated result of the script as a string, or 'Execution Blocked!'
        if the script contains dangerous actions or results in an error.
    """
    logger.info(f"Processing user code script: '{code_script}'")
    try:
        # Attempt to parse the user's script.
        # mode='eval' ensures it's a single expression.
        tree = ast.parse(code_script, mode='eval')
    except (SyntaxError, ValueError, TypeError, MemoryError) as e:
        # Catch errors during parsing (e.g., script too complex, invalid syntax).
        logger.error(f"Syntax error or parsing issue for script '{code_script}': {e}")
        return 'Execution Blocked!'

    if not _is_script_safe(tree):
        logger.warning(f"Script validation failed for script: '{code_script}'")
        return 'Execution Blocked!'

    try:
        result_str = _execute_safe_script(tree)
        logger.info(f"Successfully executed script '{code_script}'. Result: '{result_str}'")
        return result_str
    except Exception as e:
        # Catch any runtime errors during the evaluation of the (supposedly) safe code
        # from _execute_safe_script. This includes things like ZeroDivisionError,
        # OverflowError, or unexpected issues during compile() or eval().
        logger.error(f"Runtime error during execution of script '{code_script}': {e}")
        return 'Execution Blocked!'
