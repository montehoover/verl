import ast
import sys

# Define allowed AST node types for safety.
# This list determines what operations and literals are permitted.
_ALLOWED_NODE_TYPES = {
    ast.Expression,  # The top-level node for an expression.
    
    # For binary operations like +, -, *, /
    ast.BinOp,
    # For unary operations like - (negation)
    ast.UnaryOp,
    
    # Specific operator types allowed within BinOp or UnaryOp
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,    # For integer division //
    ast.Mod,         # For modulo %
    ast.Pow,         # For exponentiation **
    
    ast.USub,        # Unary minus (e.g., -5)
    ast.UAdd,        # Unary plus (e.g., +5, though often optimized out by parser)
}

# Adapt for ast.Constant (Python 3.8+) vs older literal types (Num, Str, NameConstant)
if sys.version_info >= (3, 8):
    _ALLOWED_NODE_TYPES.add(ast.Constant)  # Handles numbers, strings, None, True, False
else:
    _ALLOWED_NODE_TYPES.add(ast.Num)         # For numbers (int, float, complex)
    _ALLOWED_NODE_TYPES.add(ast.Str)         # For strings
    _ALLOWED_NODE_TYPES.add(ast.NameConstant) # For True, False, None

def _is_safe_ast_node(node):
    """
    Recursively checks if an AST node and its children are of allowed types.
    """
    # Check the current node's type
    if type(node) not in _ALLOWED_NODE_TYPES:
        return False
    
    # Recursively check all child nodes
    for child_node in ast.iter_child_nodes(node):
        if not _is_safe_ast_node(child_node):
            return False
            
    return True

def parse_user_script(script_str: str) -> str:
    """
    Safely evaluates a user-provided Python script string containing only
    simple arithmetic or string operations.

    Args:
        script_str: A string containing the Python script.

    Returns:
        The result of the script execution as a string, or 'Execution Blocked!'
        if the input is unsafe or causes an error.
    """
    try:
        # Parse the script string. mode='eval' ensures it's a single expression.
        # If script_str is not a valid expression, ast.parse will raise SyntaxError.
        tree = ast.parse(script_str, mode='eval')

        # Validate the AST to ensure it only contains allowed operations/nodes.
        # The main check is done by _is_safe_ast_node walking through the tree.
        if not _is_safe_ast_node(tree):
            return 'Execution Blocked!'

        # Compile the validated AST.
        # The filename='<user_script>' is for potential error reporting.
        compiled_code = compile(tree, filename='<user_script>', mode='eval')
        
        # Evaluate the compiled code in a heavily restricted environment.
        # '__builtins__': {} disallows access to any built-in functions.
        # {}: empty locals dictionary.
        # This means only literals and the whitelisted operators can be used.
        result = eval(compiled_code, {'__builtins__': {}}, {})
        
        return str(result)
    
    # Catch specific, common errors during parsing or evaluation.
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError, MemoryError):
        # SyntaxError: From ast.parse if script_str is malformed.
        # TypeError: E.g., 'hello' / 2.
        # ValueError: Unlikely with current restrictions (e.g. int("foo") if int() was allowed).
        # ZeroDivisionError: E.g., 1 / 0.
        # OverflowError: E.g., number too large, like 10**100000.
        # MemoryError: E.g., 'a' * (10**9) trying to create a huge string.
        return 'Execution Blocked!'
    except Exception:
        # A catch-all for any other unexpected exceptions as a safety measure.
        # This ensures that if any unhandled case arises, it's still caught safely.
        return 'Execution Blocked!'
