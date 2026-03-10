import ast

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
    if not isinstance(script_str, str):
        # Ensure input is a string
        return "Execution Blocked!"

    stripped_script = script_str.strip()
    if not stripped_script:
        # Empty script after stripping whitespace
        return "Execution Blocked!"
    
    # Basic length check to prevent extremely long inputs (e.g., > 1000 chars)
    if len(stripped_script) > 1000:
        return "Execution Blocked!"

    try:
        # Step 1: Parse the script string into an AST.
        # mode='eval' is used as we expect a single expression.
        tree = ast.parse(stripped_script, mode='eval')
    except SyntaxError:
        # Invalid Python syntax in the input string.
        return "Execution Blocked!"

    # Step 2: Validate the AST structure and its nodes.
    if not _validate_ast(tree):
        # AST validation failed (e.g., disallowed operations or nodes).
        return "Execution Blocked!"

    try:
        # Step 3: Compile the validated AST into a code object.
        code_object = _compile_ast_to_code(tree)
        
        # Step 4: Execute the compiled code object.
        result = _evaluate_compiled_code(code_object)
        
        # Step 5: Return the result as a string.
        return str(result)
    except ZeroDivisionError:
        # Specific arithmetic error caught from _evaluate_compiled_code.
        return "Execution Blocked!"
    except RuntimeError as e: 
        # Generic runtime error from _evaluate_compiled_code, indicating an issue during evaluation.
        # This also helps to ensure the message "Script evaluation failed" from the helper is not exposed.
        return "Execution Blocked!"
    except Exception:
        # Catch any other exceptions that might occur during compilation 
        # (e.g., TypeError from compile for an operation not applicable to types)
        # or other unexpected issues.
        return "Execution Blocked!"
