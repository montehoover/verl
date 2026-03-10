import ast

# Whitelist of AST node types considered safe for this execution environment.
# This list restricts scripts to basic data types, arithmetic, assignments,
# and simple data structure manipulations. It notably excludes loops,
# function definitions, imports, attribute access, and direct function calls.
ALLOWED_NODE_TYPES = {
    ast.Module, ast.Expr, ast.Constant, ast.Name, ast.Load, ast.Store,
    ast.Assign, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
    # Operators for BinOp
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    # Operators for UnaryOp
    ast.UAdd, ast.USub, ast.Not,
    # Operators for Compare
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
    # Operators for BoolOp
    ast.And, ast.Or,
    # Allow simple data structures: lists, tuples, dicts, sets
    ast.List, ast.Tuple, ast.Dict, ast.Set,
    # Allow subscripting (e.g., my_list[0]) and slicing (e.g., my_list[1:3])
    ast.Subscript, ast.Slice,
    # Note: ast.Index is deprecated in Python 3.9+; indices are ast.Constant or other expressions.
}

# Whitelist of built-in functions/constants accessible to the script.
# These are provided in the __builtins__ dictionary for the executed code.
ALLOWED_BUILTINS = {
    'abs': abs, 'all': all, 'any': any, 'bool': bool, 'int': int, 'float': float,
    'str': str, 'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
    'len': len, 'max': max, 'min': min, 'pow': pow,
    'range': range, 'round': round, 'sorted': sorted, 'sum': sum,
    'True': True, 'False': False, 'None': None,
}

# Prohibited names that cannot be used as variable names or accessed.
PROHIBITED_NAMES = {
    '__import__', 'eval', 'exec', 'open', 'compile', 'getattr', 'setattr',
    'delattr', 'globals', 'locals', 'vars', '__builtins__', '__file__',
    '__name__', '__package__', '__doc__', '__spec__', '__loader__',
    '__cached__', '__annotations__'
}


class AstValidator(ast.NodeVisitor):
    """
    A node visitor to traverse the AST and ensure all nodes are of allowed types
    and that certain dangerous names are not used.
    """
    def visit(self, node):
        node_type = type(node)
        if node_type not in ALLOWED_NODE_TYPES:
            raise ValueError(
                f"Prohibited operation: AST node type {node_type.__name__} is not allowed."
            )

        if isinstance(node, ast.Name):
            if node.id in PROHIBITED_NAMES:
                raise ValueError(
                    f"Prohibited operation: Usage of reserved or unsafe name '{node.id}'."
                )
        
        # Recursively visit child nodes
        super().visit(node)


def execute_python_script(python_code: str):
    """
    Executes a user-provided Python script in a restricted environment.

    Args:
        python_code: A string containing the Python code to execute.

    Returns:
        The result of the last expression statement in the script, 
        or None if the script does not end with an expression statement
        or produces no specific result.

    Raises:
        ValueError: If the script contains invalid syntax, uses prohibited
                    operations, or encounters an error during execution.
    """
    try:
        # Parse the Python code into an AST
        tree = ast.parse(python_code, mode='exec')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")

    # Validate the AST
    validator = AstValidator()
    try:
        validator.visit(tree)
    except ValueError:
        raise  # Re-raise validation errors directly

    # Compile the AST to bytecode
    try:
        # The <user_script> filename is conventional for code from a string
        code_obj = compile(tree, filename="<user_script>", mode="exec")
    except Exception as e:
        # Catch potential errors during compilation after AST validation
        raise ValueError(f"Error compiling script: {e}")

    # Prepare restricted globals and an empty locals dictionary for execution
    restricted_globals = {"__builtins__": ALLOWED_BUILTINS}
    script_locals = {}

    # Execute the bytecode
    try:
        exec(code_obj, restricted_globals, script_locals)
    except Exception as e:
        # Catch runtime errors from the executed script (e.g., NameError, TypeError)
        raise ValueError(f"Error during script execution: {type(e).__name__}: {e}")

    # Determine the result: if the last statement was an expression, evaluate it.
    result = None
    if tree.body:
        last_statement = tree.body[-1]
        if isinstance(last_statement, ast.Expr):
            # The 'value' of an ast.Expr node is the actual expression node.
            result_expr_node = last_statement.value
            
            # Compile this expression in 'eval' mode.
            # ast.Expression is a wrapper suitable for eval mode.
            eval_tree = ast.Expression(body=result_expr_node)
            ast.fix_missing_locations(eval_tree) # Important for compile

            try:
                eval_code_obj = compile(eval_tree, filename="<user_script_last_expr>", mode="eval")
                result = eval(eval_code_obj, restricted_globals, script_locals)
            except Exception as e:
                # If evaluating the final expression causes an error.
                raise ValueError(f"Error evaluating result expression: {type(e).__name__}: {e}")
    
    return result
