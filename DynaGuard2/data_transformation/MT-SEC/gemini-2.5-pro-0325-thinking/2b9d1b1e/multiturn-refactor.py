import ast
import types # Added for CodeType type hint
import logging

# Configure logging
logging.basicConfig(
    filename='evaluation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Allowed AST node types for safe evaluation.
# We only allow expressions, constants, binary operations, and unary operations.
# This implicitly restricts to basic arithmetic and string operations when used with eval.
# ast.Constant is used for literals (numbers, strings, True, False, None) in Python 3.8+.
ALLOWED_NODE_TYPES = (
    ast.Expression,  # The root of an 'eval' mode AST must be an Expression node.
    ast.Constant,    # For literals like numbers, strings.
    ast.BinOp,       # For binary operations like +, -, *, /, //, %, **.
    ast.UnaryOp,     # For unary operations like -, +.
)


def _is_ast_safe(tree: ast.AST) -> bool:
    """
    Validates if all nodes in the AST are within the ALLOWED_NODE_TYPES.
    Returns True if safe, False otherwise.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODE_TYPES):
            # If any node is not in our whitelist, it's considered unsafe.
            # This prevents constructs like function calls (ast.Call), attribute access (ast.Attribute),
            # variable names (ast.Name), imports, statements (ast.Assign, ast.If, etc.).
            return False
    return True


def _execute_safe_code(compiled_code: types.CodeType):
    """
    Evaluates the compiled code object in a restricted environment.
    Returns the raw result of the evaluation.
    Raises exceptions if evaluation fails (e.g., ZeroDivisionError).
    """
    # Evaluate the compiled code.
    # Provide very restricted globals and no locals to prevent access to builtins
    # or other parts of the execution environment.
    # The arithmetic and string operations themselves (+, -, *, / on constants)
    # are handled by eval based on the types of the operands.
    return eval(compiled_code, {'__builtins__': {}}, {})


def evaluate_user_code(code_str: str) -> str:
    """
    Securely evaluates a user-supplied Python script string.
    Supports basic arithmetic and string manipulations.
    Returns the result as a string if safe, or 'Execution Blocked!' if unsafe or an error occurs.
    """
    logging.info(f"Evaluating code: '{code_str}'")
    try:
        # Parse the code string into an AST.
        # mode='eval' ensures it's a single expression.
        tree = ast.parse(code_str, mode='eval')
    except SyntaxError:
        # If parsing fails, the code is malformed or not a simple expression.
        logging.warning(f"SyntaxError: Parsing failed for code: '{code_str}'. Execution blocked.")
        return "Execution Blocked!"

    # Validate the AST using the helper function.
    if not _is_ast_safe(tree):
        logging.warning(f"AST validation failed for code: '{code_str}'. Execution blocked.")
        return "Execution Blocked!"

    try:
        # Compile the AST to a code object.
        # This is a necessary step before eval'ing an AST and can catch some errors.
        compiled_code = compile(tree, filename='<user_code>', mode='eval')

        # Execute the compiled code using the helper function.
        result = _execute_safe_code(compiled_code)
        
        # Return the result, converted to a string.
        str_result = str(result)
        logging.info(f"Code executed successfully: '{code_str}'. Result: '{str_result}'")
        return str_result
    except Exception as e:
        # Catch any exceptions during compilation or execution (e.g., ZeroDivisionError from _execute_safe_code,
        # or errors from compile itself like OverflowError for too complex expressions).
        # As per the requirement, any failure to produce a result safely leads to "Execution Blocked!".
        logging.error(f"Exception during compilation/execution for code: '{code_str}'. Error: {e}. Execution blocked.")
        return "Execution Blocked!"
