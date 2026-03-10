import ast
import logging

# Configure a logger for this module
logger = logging.getLogger(__name__)

# Helper function to recursively check AST nodes
def _is_safe_node_for_parse_user_script(node):
    """
    Recursively checks if an AST node is safe for evaluation.
    Allowed nodes: Constant (numbers, strings, booleans),
                   BinOp (specific arithmetic/string ops),
                   UnaryOp (specific arithmetic ops).
    """
    if isinstance(node, ast.Constant):
        # Allow numbers (int, float, complex), strings, and booleans.
        # None is disallowed as it's not typically part of "arithmetic/string operations".
        return isinstance(node.value, (int, float, complex, str, bool))
    elif isinstance(node, ast.BinOp):
        # Allowed binary operators: +, -, *, /, %, **
        # String concatenation uses ast.Add.
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)):
            return False
        # Recursively check operands
        return _is_safe_node_for_parse_user_script(node.left) and \
               _is_safe_node_for_parse_user_script(node.right)
    elif isinstance(node, ast.UnaryOp):
        # Allowed unary operators: -, + (arithmetic negation and identity)
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            return False
        # Recursively check operand
        return _is_safe_node_for_parse_user_script(node.operand)
    
    # Any other node type is considered unsafe
    return False

def _validate_ast(tree: ast.AST) -> bool:
    """
    Validates the overall structure of the parsed AST.
    Ensures it's an Expression and its body consists of safe nodes
    using _is_safe_node_for_parse_user_script.
    """
    if not isinstance(tree, ast.Expression):
        return False
    return _is_safe_node_for_parse_user_script(tree.body)

def _compile_ast_to_code(tree: ast.AST):
    """
    Compiles a validated AST to a code object.
    Filename is set to '<user_script>' and mode to 'eval'.
    Can raise exceptions if compilation fails (e.g., TypeError for invalid operations).
    """
    return compile(tree, filename='<user_script>', mode='eval')

def _evaluate_compiled_code(code_object):
    """
    Evaluates a compiled code object in a highly restricted environment.
    Raises ZeroDivisionError directly if it occurs.
    Wraps other evaluation-time exceptions in a RuntimeError.
    """
    try:
        # __builtins__: {} removes access to all built-in functions and names.
        # {}: provides an empty dictionary for local variables.
        return eval(code_object, {'__builtins__': {}}, {})
    except ZeroDivisionError:
        raise  # Re-raise to be specifically caught by the caller
    except Exception as e:
        # Wrap other runtime errors during evaluation in a generic error.
        # This helps distinguish evaluation phase errors from compilation or parsing errors.
        raise RuntimeError("Script evaluation failed") from e

def parse_user_script(script_str: str) -> str:
    """
    Safely evaluates a user-provided Python script string containing only simple
    arithmetic or string operations.

    Args:
        script_str: A string containing a tiny Python script.

    Returns:
        The result of the script execution as a string, or 'Execution Blocked!'
        if the input is unsafe or causes an error.
    """
    # Truncate script for logging if it's too long
    script_for_log = script_str[:200] + "..." if isinstance(script_str, str) and len(script_str) > 200 else script_str
    logger.info(f"Attempting to parse script: '{script_for_log}'")

    if not isinstance(script_str, str):
        logger.warning("Input script_str is not a string. Blocking execution.")
        return "Execution Blocked!"

    stripped_script = script_str.strip()
    if not stripped_script:
        logger.warning("Input script is empty after stripping. Blocking execution.")
        return "Execution Blocked!"
    
    # Basic length check to prevent extremely long inputs (e.g., > 1000 chars)
    if len(stripped_script) > 1000:
        logger.warning(f"Input script exceeds maximum length ({len(stripped_script)} > 1000). Blocking execution.")
        return "Execution Blocked!"

    try:
        # Step 1: Parse the script string into an AST.
        # mode='eval' is used as we expect a single expression.
        logger.debug("Parsing script string into AST.")
        tree = ast.parse(stripped_script, mode='eval')
    except SyntaxError as e:
        logger.error(f"SyntaxError during script parsing: {e}. Script: '{script_for_log}'. Blocking execution.")
        return "Execution Blocked!"

    # Step 2: Validate the AST structure and its nodes.
    logger.debug("Validating AST structure.")
    if not _validate_ast(tree):
        logger.warning(f"AST validation failed for script: '{script_for_log}'. Blocking execution.")
        return "Execution Blocked!"

    try:
        # Step 3: Compile the validated AST into a code object.
        logger.debug("Compiling AST to code object.")
        code_object = _compile_ast_to_code(tree)
        
        # Step 4: Execute the compiled code object.
        logger.debug("Evaluating compiled code object.")
        result = _evaluate_compiled_code(code_object)
        
        # Step 5: Return the result as a string.
        result_str = str(result)
        logger.info(f"Script successfully executed. Script: '{script_for_log}'. Result: '{result_str}'")
        return result_str
    except ZeroDivisionError:
        logger.error(f"ZeroDivisionError during script execution. Script: '{script_for_log}'. Blocking execution.")
        return "Execution Blocked!"
    except RuntimeError as e: 
        logger.error(f"RuntimeError during script evaluation: {e}. Script: '{script_for_log}'. Blocking execution.")
        return "Execution Blocked!"
    except Exception as e:
        # Catch any other exceptions that might occur during compilation or other unexpected issues.
        logger.error(f"Unexpected exception during script processing: {e}. Script: '{script_for_log}'. Blocking execution.")
        return "Execution Blocked!"
