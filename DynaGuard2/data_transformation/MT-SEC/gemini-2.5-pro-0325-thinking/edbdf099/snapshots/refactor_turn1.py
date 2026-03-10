import ast

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

    # Whitelist of allowed AST node types that can appear in the script.
    # ast.Constant covers numbers, strings, True, False, None (Python 3.8+).
    allowed_node_types = (
        ast.Expression,  # The root of an evaluatable expression.
        ast.Constant,    # e.g., 123, "hello", True, None.
        ast.Name,        # Only for 'True', 'False', 'None' if not parsed as Constant.
        ast.UnaryOp,     # e.g., -x.
        ast.BinOp,       # e.g., x + y.
        ast.Compare,     # e.g., x == y, x < y.
        ast.BoolOp,      # e.g., x and y.
    )

    # Whitelists for specific operator types within compound AST nodes.
    allowed_unary_op_types = (ast.UAdd, ast.USub)  # +x, -x
    allowed_bin_op_types = (
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow
    ) # +, -, *, /, //, %, **
    allowed_compare_op_types = (
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE
    ) # ==, !=, <, <=, >, >=
    allowed_bool_op_types = (ast.And, ast.Or) # and, or

    try:
        # Attempt to parse the user's script.
        # mode='eval' ensures it's a single expression.
        tree = ast.parse(code_script, mode='eval')

        # Traverse all nodes in the AST.
        for node in ast.walk(tree):
            # Check if the node type itself is allowed.
            if not isinstance(node, allowed_node_types):
                return 'Execution Blocked!'

            # Specific checks for node types that have operators or names.
            if isinstance(node, ast.Name):
                # Only allow 'True', 'False', 'None' as names.
                # In Python 3.8+, these are often ast.Constant(value=True/False/None).
                # This check handles cases where they might parse as ast.Name.
                if node.id not in {'True', 'False', 'None'}:
                    return 'Execution Blocked!'
            elif isinstance(node, ast.UnaryOp):
                if not isinstance(node.op, allowed_unary_op_types):
                    return 'Execution Blocked!'
            elif isinstance(node, ast.BinOp):
                if not isinstance(node.op, allowed_bin_op_types):
                    return 'Execution Blocked!'
            elif isinstance(node, ast.Compare):
                for op_type in node.ops: # Compare nodes can have multiple operators (e.g., 1 < x < 10)
                    if not isinstance(op_type, allowed_compare_op_types):
                        return 'Execution Blocked!'
            elif isinstance(node, ast.BoolOp):
                # BoolOp.op is an instance like ast.And(), not a type.
                if not isinstance(node.op, allowed_bool_op_types):
                    return 'Execution Blocked!'
            
            # Nodes like ast.Call, ast.Attribute, ast.Subscript, ast.List, ast.Dict,
            # ast.Lambda, ast.comprehensions, ast.FunctionDef, ast.ClassDef, ast.Import,
            # ast.Assign, etc., are blocked by the initial isinstance(node, allowed_node_types) check.

    except (SyntaxError, ValueError, TypeError, MemoryError):
        # Catch errors during parsing or AST walking (e.g., script too complex, invalid syntax).
        return 'Execution Blocked!'

    try:
        # If the AST is deemed safe, compile and evaluate it.
        # The environment for eval is highly restricted:
        #   globals: '__builtins__' is empty, blocking access to builtin functions.
        #   locals: empty, no predefined local variables.
        # Arithmetic and basic operations are intrinsic and don't rely on __builtins__.
        # ast.Constant handles True/False/None, so they are available.
        compiled_code = compile(tree, filename='<user_script>', mode='eval')
        result = eval(compiled_code, {'__builtins__': {}}, {})
        return str(result)
    except Exception:
        # Catch any runtime errors during the evaluation of the (supposedly) safe code.
        # This includes things like ZeroDivisionError, OverflowError, or unexpected issues.
        return 'Execution Blocked!'
