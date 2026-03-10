import ast

# Whitelist of allowed AST node types
# (Includes types for Python <3.8 and >=3.8 constants for broader compatibility)
ALLOWED_NODES = {
    ast.Module, ast.Expr,
    ast.Load, ast.Store, ast.Del,  # Contexts for names
    ast.Constant, ast.Num, ast.Str, ast.Bytes, ast.NameConstant, # Literals (NameConstant for True/False/None)
    ast.Name,  # Variables
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,  # Operators
    ast.List, ast.Tuple, ast.Dict, ast.Set,  # Data structures
    ast.Assign, ast.AugAssign,  # Assignments
    ast.If, ast.For, ast.While,  # Control Flow
    ast.Break, ast.Continue, ast.Pass,  # Control Flow statements
    ast.Subscript, ast.Slice, ast.Index, # Index for e.g. list[0], Slice for list[0:1]
    ast.Call,  # Function calls
    ast.Delete, # For 'del x' statement
    # Explicitly disallowed (by omission): ast.Attribute, ast.FunctionDef, ast.ClassDef, ast.Lambda,
    # ast.Import, ast.ImportFrom, ast.Try, ast.With, ast.Raise, ast.Assert, ast.Global, ast.Nonlocal,
    # ast.Yield, ast.Await, ast.Starred (e.g. *args), etc.
}

# Whitelist of allowed built-in function names (as strings)
ALLOWED_BUILTIN_NAMES = {
    'print', 'len', 'abs', 'round', 'min', 'max', 'sum', 'str', 'int', 'float', 'bool',
    'list', 'dict', 'tuple', 'set', 'range', 'zip', 'enumerate', 'sorted', 'reversed',
    'all', 'any',
    # 'eval', 'exec', 'open', 'input', '__import__' are dangerous and intentionally excluded.
}


class ScriptValidator(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.errors = []

    def visit(self, node):
        node_type = type(node)
        if node_type not in ALLOWED_NODES:
            self.errors.append(f"Disallowed operation: {node_type.__name__} at line {getattr(node, 'lineno', 'N/A')}")
            return  # Stop validation for this branch if a node type is disallowed.
        super().visit(node)  # Continue visiting children

    def visit_Call(self, node):
        # This method is called after the main visit() has confirmed ast.Call is an allowed node type.
        # Now, check the function being called.
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in ALLOWED_BUILTIN_NAMES:
                self.errors.append(f"Disallowed function call: {func_name} at line {node.lineno}")
        else:
            # node.func is not a simple name (e.g., could be a lambda if it were allowed, or an attribute).
            # Since ast.Attribute and ast.Lambda are not in ALLOWED_NODES, they would be caught by the main `visit` method
            # if they appeared as the function to be called.
            # This error is for other unexpected callable types that might form node.func.
            self.errors.append(f"Disallowed callable expression type: {type(node.func).__name__} at line {node.lineno}")
        
        self.generic_visit(node) # Ensure arguments and other parts of the call are visited.


def _parse_and_validate_script(user_script: str):
    """
    Parses and validates the user script.

    Args:
        user_script: The Python script string.

    Returns:
        The parsed and validated AST tree.

    Raises:
        ValueError: If syntax errors or disallowed operations are found.
    """
    try:
        tree = ast.parse(user_script, mode='exec')
    except SyntaxError as e:
        raise ValueError(f"Syntax error in user script: {e}") from e

    validator = ScriptValidator()
    validator.visit(tree)

    if validator.errors:
        unique_errors = sorted(list(set(validator.errors)))
        raise ValueError("User script contains disallowed operations:\n" + "\n".join(unique_errors))
    
    return tree


def run_user_script(user_script: str):
    """
    Executes a Python script supplied by the user, limited to safe operations.

    Args:
        user_script: str, the Python script provided by the user.

    Returns:
        The result of the script if any, or None.

    Raises:
        ValueError: If the script contains disallowed operations or syntax errors.
    """
    tree = _parse_and_validate_script(user_script)

    # Prepare a safe environment for execution
    safe_builtins_for_exec = {}
    for name in ALLOWED_BUILTIN_NAMES:
        try:
            # __builtins__ is typically a reference to the builtins module in the current scope
            safe_builtins_for_exec[name] = getattr(__builtins__, name)
        except AttributeError:
            # This case should ideally not happen if ALLOWED_BUILTIN_NAMES is curated correctly
            # and __builtins__ is standard.
            # Consider logging a warning or raising a configuration error if this occurs.
            pass 
    
    safe_globals = {'__builtins__': safe_builtins_for_exec}
    # No other globals are provided by default.

    script_locals = {}
    result_var_name = "__script_result__"

    # If the script's last statement is an expression, modify the AST to capture its result.
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        # Create an assignment node: __script_result__ = <original_last_expr_value>
        assign_target = ast.Name(id=result_var_name, ctx=ast.Store())
        assign_node = ast.Assign(targets=[assign_target], value=tree.body[-1].value)
        
        # Copy line numbers from the original expression to the new assignment statement
        ast.copy_location(assign_node, tree.body[-1])
        ast.fix_missing_locations(assign_node) # Crucial for compile()

        tree.body[-1] = assign_node
    
    try:
        # Compile the potentially modified AST
        code_obj = compile(tree, filename="<user_script>", mode="exec")
    except Exception as e:
        # Catch potential errors during compilation (e.g., from AST manipulation)
        raise ValueError(f"Error compiling script: {e}") from e

    try:
        exec(code_obj, safe_globals, script_locals)
    except Exception as e:
        # Catch runtime errors from the script itself
        raise ValueError(f"Error during script execution: {type(e).__name__}: {e}") from e

    return script_locals.get(result_var_name, None)
