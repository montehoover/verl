import ast

# Visitor class to inspect AST nodes for disallowed operations
class SafeCodeVisitor(ast.NodeVisitor):
    """
    Inspects AST nodes to ensure code safety by disallowing certain operations.
    Currently disallows:
    - Import statements ('import ...' and 'from ... import ...').
    - Direct calls to specified functions (e.g., 'eval()', 'exec()', 'open()', '__import__()').
    """
    # Set of function names that are disallowed when called directly
    DISALLOWED_CALLS = {'eval', 'exec', 'open', '__import__'}

    def visit_Import(self, node):
        """Disallow 'import ...' statements."""
        raise ValueError("Imports are not allowed.")

    def visit_ImportFrom(self, node):
        """Disallow 'from ... import ...' statements."""
        raise ValueError("Imports (from) are not allowed.")

    def visit_Call(self, node):
        """
        Disallow calls to certain functions (e.g., eval, exec, open).
        This checks for direct calls like 'eval(...)' or 'open(...)'.
        Aliasing (e.g., x = open; x()) is not covered by this basic check.
        """
        if isinstance(node.func, ast.Name):
            if node.func.id in self.DISALLOWED_CALLS:
                raise ValueError(f"Call to disallowed function '{node.func.id}' is not allowed.")
        
        # To further enhance security, one might add checks for attribute calls, e.g.:
        # if isinstance(node.func, ast.Attribute):
        #     # Example: block 'os.system()'
        #     if isinstance(node.func.value, ast.Name) and node.func.value.id == 'os' \
        #        and node.func.attr == 'system':
        #         raise ValueError("Calls to 'os.system' are not allowed.")
        
        # Ensure all child nodes are also visited to catch nested disallowed operations
        self.generic_visit(node)


def _parse_and_validate_code(code_input: str) -> ast.AST:
    """
    Parses the Python code string into an AST and validates it for safety.

    Args:
        code_input: The Python code string.

    Returns:
        The validated Abstract Syntax Tree.

    Raises:
        ValueError: If syntax is invalid or disallowed operations are found.
    """
    try:
        # Parse the code into an Abstract Syntax Tree (AST)
        tree = ast.parse(code_input, filename='<user_code>')
    except SyntaxError as e:
        # If parsing fails, it's a syntax error in the user's code
        raise ValueError(f"Syntax error in code: {e}")

    # Validate the AST for disallowed operations
    validator = SafeCodeVisitor()
    try:
        validator.visit(tree)
    except ValueError: # Catch ValueError from our validator
        raise # Re-raise it as it's the specified error type
    return tree


def _execute_validated_code(tree: ast.AST):
    """
    Executes the validated AST.

    Args:
        tree: The validated Abstract Syntax Tree.

    Returns:
        The result of the execution, or None.

    Raises:
        ValueError: If there's an error compiling the AST.
        Other runtime errors from the executed code may propagate.
    """
    # Prepare execution environment.
    # Using default __builtins__ here. For higher security, a curated dictionary
    # of safe builtins should be provided in exec_globals.
    # e.g., safe_builtins = {'print': print, 'len': len, 'str': str, ...}
    # exec_globals = {"__builtins__": safe_builtins}
    exec_globals = {"__builtins__": __builtins__}
    exec_locals = {}

    result = None
    # If the AST module's body is not empty and the last statement is an ast.Expr,
    # we will execute preceding statements and then evaluate this final expression.
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        # Get the last expression statement node
        last_expr_stmt_node = tree.body.pop() # Modifies the tree

        # Compile and execute the preceding statements (if any)
        if tree.body:
            # Create a module AST for the statements part
            statements_module = ast.Module(body=tree.body, type_ignores=[])
            try:
                # Compile the statements part of the code
                code_obj_stmts = compile(statements_module, filename='<user_code_stmts>', mode='exec')
            except Exception as e: # Catch potential compilation errors for partial AST
                raise ValueError(f"Error compiling statements: {e}")

            # Execute the compiled statements
            # Runtime errors from user code will propagate from here
            exec(code_obj_stmts, exec_globals, exec_locals)

        # Now, compile and evaluate the last expression.
        # ast.Expression node is used for 'eval' mode compilation.
        # Its 'body' should be the actual expression value from the ast.Expr node.
        eval_wrapper_node = ast.Expression(body=last_expr_stmt_node.value)
        try:
            code_obj_expr = compile(eval_wrapper_node, filename='<user_code_expr>', mode='eval')
        except Exception as e: # Catch potential compilation errors for the expression part
            raise ValueError(f"Error compiling final expression: {e}")

        # Evaluate the compiled expression
        # Runtime errors from user expression will propagate from here
        result = eval(code_obj_expr, exec_globals, exec_locals)

    elif tree.body: # Code has statements but doesn't end with an expression, or is empty.
        # Just execute the whole module.
        # Note: tree might have been modified if it ended with an Expr and also had prior statements.
        # However, if it didn't end with Expr, tree.body is as original.
        # If it did end with Expr but had no prior statements, tree.body is now empty,
        # and this 'elif' block won't be hit.
        # This logic path is for when the original code did not end in an expression.
        try:
            # We need to compile the original tree if we are in this block,
            # but ast.parse(code_input) would be reparsing.
            # The tree passed here is already potentially modified (last Expr popped).
            # This needs careful handling.
            # For simplicity and correctness, if we reach here, it means the original tree
            # did not end in an ast.Expr, so the 'tree' object is safe to compile as 'exec'.
            code_obj = compile(tree, filename='<user_code>', mode='exec')
        except Exception as e: # Catch potential compilation errors
            raise ValueError(f"Error compiling code: {e}")

        # Execute the compiled code
        # Runtime errors from user code will propagate from here
        exec(code_obj, exec_globals, exec_locals)
        # result remains None as 'exec' doesn't return a value in this way

    # If tree.body is empty (e.g. empty string or only comments originally, or after popping Expr), result is None.
    return result


def process_code_input(code_input: str):
    """
    Executes a Python code snippet with safety restrictions enforced by AST analysis.

    Args:
        code_input: A string containing the Python code to execute.

    Returns:
        The result of the last expression in the code snippet, if it was an expression.
        Returns None if the code snippet does not end with an expression (e.g., ends
        with an assignment or definition), if the snippet is empty, or if it only
        contains statements that don't produce a returnable value in this context.

    Raises:
        TypeError: If code_input is not a string.
        ValueError: If the code_input contains syntax errors, or if it attempts
                    to perform disallowed operations (e.g., imports, calls to
                    functions like eval, exec, open), or compilation errors.
        Any other exceptions raised by the user's code during execution (if the
        code is deemed safe by AST analysis but fails at runtime) will propagate.
    """
    if not isinstance(code_input, str):
        raise TypeError("code_input must be a string.")

    # Note: _parse_and_validate_code can raise ValueError
    tree = _parse_and_validate_code(code_input)
    
    # Note: _execute_validated_code can raise ValueError or other runtime errors
    # A copy of the tree's body might be needed if _execute_validated_code modifies it
    # and the original structure is important for different execution paths.
    # ast.parse returns a new tree object each time.
    # The current _execute_validated_code modifies tree.body by popping.
    # To make _execute_validated_code purer, it should not modify its input 'tree'.
    # Let's adjust _execute_validated_code to work with a copy or handle this.
    # For now, we pass the tree, and it's modified. This is acceptable if _execute_validated_code
    # is only called once with this specific tree instance.

    # The docstring of _execute_validated_code mentions it might modify the tree.
    # Let's make it explicit that it operates on a potentially modified tree.
    # The current logic in _execute_validated_code handles the modification correctly
    # by checking tree.body *after* the pop.

    result = _execute_validated_code(tree)
    return result
