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
        # Parse the script string. mode='eval' is crucial as we expect an expression.
        # This will raise SyntaxError for statements or invalid Python syntax.
        tree = ast.parse(stripped_script, mode='eval')
    except SyntaxError:
        return "Execution Blocked!" # Invalid Python syntax

    # The root of an 'eval' mode AST is an Expression node.
    # Its 'body' contains the actual expression to be evaluated.
    # Validate the AST structure starting from the body of the expression.
    if not isinstance(tree, ast.Expression) or \
       not _is_safe_node_for_parse_user_script(tree.body):
        return "Execution Blocked!" # AST validation failed (unsafe node detected)

    try:
        # Compile the AST into a code object. This is a prerequisite for eval if input is AST.
        # filename='<user_script>' is a convention for dynamically generated code.
        code_object = compile(tree, filename='<user_script>', mode='eval')
        
        # Evaluate the compiled code object in a highly restricted environment.
        # __builtins__: {} removes access to all built-in functions and names.
        # {}: provides an empty dictionary for local variables (no user-defined vars).
        result = eval(code_object, {'__builtins__': {}}, {})
        
        # Return the result converted to a string.
        return str(result)
    except ZeroDivisionError:
        # Specifically catch division by zero as an "issue with the script".
        return "Execution Blocked!"
    except Exception:
        # Catch any other runtime errors during evaluation (e.g., OverflowError, TypeError).
        # These are also considered "issues with the script".
        return "Execution Blocked!"
