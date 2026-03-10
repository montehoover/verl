import ast

# Whitelist of safe function names that can be called from the script
SAFE_FUNCTION_CALLS = {
    'print', 'len', 'int', 'float', 'str', 'list', 'dict', 'tuple', 'set',
    'range', 'abs', 'round', 'sum', 'min', 'max', 'sorted', 'all', 'any',
    'isinstance', 'issubclass', # Generally safe for type checking
    'map', 'filter', 'zip',     # Functional constructs
}

# Dictionary of built-ins to expose to the executed script
# Ensure this is consistent with SAFE_FUNCTION_CALLS
SAFE_BUILTINS = {
    name: __builtins__[name] for name in SAFE_FUNCTION_CALLS if name in __builtins__
}
# Add essential constants like True, False, None if not covered by SAFE_FUNCTION_CALLS
SAFE_BUILTINS.update({'True': True, 'False': False, 'None': None})


class SafeASTVisitor(ast.NodeVisitor):
    """
    A node visitor to ensure only whitelisted AST nodes and operations are used.
    """
    def __init__(self):
        super().__init__()
        self.allowed_call_names = SAFE_FUNCTION_CALLS

    def generic_visit(self, node):
        """
        Fallback for any AST node type not explicitly handled by a visit_NodeName method.
        Disallows any unhandled node type.
        """
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

    # --- Disallowed node types ---
    def visit_Import(self, node): raise ValueError("Import statements are not allowed.")
    def visit_ImportFrom(self, node): raise ValueError("Import statements (from ... import ...) are not allowed.")
    def visit_Exec(self, node): raise ValueError("Exec statements are not allowed.")
    def visit_Try(self, node): raise ValueError("Try/except/finally blocks are not allowed.")
    def visit_With(self, node): raise ValueError("With statements are not allowed.")
    def visit_Raise(self, node): raise ValueError("Raise statements are not allowed.")
    def visit_Assert(self, node): raise ValueError("Assert statements are not allowed.")
    def visit_Global(self, node): raise ValueError("Global keyword is not allowed.")
    def visit_Nonlocal(self, node): raise ValueError("Nonlocal keyword is not allowed.")
    def visit_ClassDef(self, node): raise ValueError("Class definitions are not allowed.")
    # Async, Match, Yield, Await, FormattedValue, JoinedStr etc. will be caught by generic_visit

    # --- Potentially dangerous, handled with specific logic ---
    def visit_Attribute(self, node):
        # Disallow all attribute access for maximum safety initially.
        # This means methods like 'str'.upper() or list.append() are not allowed.
        raise ValueError("Attribute access is not allowed.")

    def visit_Delete(self, node):
        # Disallow delete operations for now.
        ast.NodeVisitor.generic_visit(self, node) # Visit targets to ensure they are simple if allowed later
        raise ValueError("Delete statements are not allowed.")

    def visit_Call(self, node):
        # First, visit the callable itself (node.func).
        # This will trigger visit_Name, visit_Attribute, etc.
        # If node.func is an ast.Attribute, visit_Attribute will raise an error.
        self.visit(node.func)

        if not isinstance(node.func, ast.Name):
            # If self.visit(node.func) didn't raise (e.g. if node.func was a complex expr that resolved to a callable),
            # we still restrict calls to named functions only.
            raise ValueError("Callable must be a direct whitelisted function name.")

        func_name = node.func.id
        if func_name not in self.allowed_call_names:
            raise ValueError(f"Calling disallowed function: {func_name}")

        # Visit arguments and keyword arguments
        for arg_node in node.args:
            self.visit(arg_node)
        for kw_node in node.keywords:
            self.visit(kw_node.value) # ast.keyword's value is the expression

    # --- Allowed node types that have children ---
    # For these, we use ast.NodeVisitor.generic_visit to recurse.
    # This ensures their children are also visited by this SafeASTVisitor.

    def visit_Module(self, node): ast.NodeVisitor.generic_visit(self, node)
    def visit_Expr(self, node): ast.NodeVisitor.generic_visit(self, node)
    def visit_Assign(self, node): ast.NodeVisitor.generic_visit(self, node)
    def visit_AugAssign(self, node): ast.NodeVisitor.generic_visit(self, node)
    def visit_Return(self, node): ast.NodeVisitor.generic_visit(self, node)
    def visit_If(self, node): ast.NodeVisitor.generic_visit(self, node) # test, body, orelse
    def visit_For(self, node): ast.NodeVisitor.generic_visit(self, node) # target, iter, body, orelse
    def visit_While(self, node): ast.NodeVisitor.generic_visit(self, node) # test, body, orelse
    def visit_Compare(self, node): ast.NodeVisitor.generic_visit(self, node) # left, ops, comparators
    def visit_BinOp(self, node): ast.NodeVisitor.generic_visit(self, node) # left, op, right
    def visit_UnaryOp(self, node): ast.NodeVisitor.generic_visit(self, node) # op, operand
    def visit_BoolOp(self, node): ast.NodeVisitor.generic_visit(self, node) # op, values
    def visit_Subscript(self, node): ast.NodeVisitor.generic_visit(self, node) # value, slice
    def visit_List(self, node): ast.NodeVisitor.generic_visit(self, node) # elts
    def visit_Tuple(self, node): ast.NodeVisitor.generic_visit(self, node) # elts
    def visit_Dict(self, node): ast.NodeVisitor.generic_visit(self, node) # keys, values
    def visit_Set(self, node): ast.NodeVisitor.generic_visit(self, node) # elts
    def visit_ListComp(self, node): ast.NodeVisitor.generic_visit(self, node) # elt, generators
    def visit_SetComp(self, node): ast.NodeVisitor.generic_visit(self, node) # elt, generators
    def visit_DictComp(self, node): ast.NodeVisitor.generic_visit(self, node) # key, value, generators
    def visit_GeneratorExp(self, node): ast.NodeVisitor.generic_visit(self, node) # elt, generators
    def visit_comprehension(self, node): ast.NodeVisitor.generic_visit(self, node) # target, iter, ifs, is_async

    def visit_FunctionDef(self, node):
        if node.name in self.allowed_call_names: # Prevent shadowing safe builtins
            raise ValueError(f"Function definition '{node.name}' shadows a protected name.")
        # Check for type hints if they are complex expressions (not typical)
        if node.returns:
            self.visit(node.returns)
        for dec in node.decorator_list: # Disallow decorators for simplicity
             raise ValueError("Decorators are not allowed.")
        self.visit(node.args) # Visit arguments (ast.arguments node)
        for stmt in node.body: # Visit body statements
            self.visit(stmt)

    def visit_Lambda(self, node):
        self.visit(node.args) # Visit arguments (ast.arguments node)
        self.visit(node.body) # Visit lambda body expression

    def visit_arguments(self, node): # For FunctionDef and Lambda args
        for arg_node in node.posonlyargs: self.visit(arg_node)
        for arg_node in node.args: self.visit(arg_node)
        if node.vararg: self.visit(node.vararg)
        for arg_node in node.kwonlyargs: self.visit(arg_node)
        # Defaults and kw_defaults are expressions
        for def_expr in node.defaults: self.visit(def_expr)
        for def_expr in node.kw_defaults:
            if def_expr: self.visit(def_expr)
            
    def visit_arg(self, node): # For arguments in ast.arguments
        if node.annotation: # Disallow complex annotations
            self.visit(node.annotation)


    def visit_IfExp(self, node): ast.NodeVisitor.generic_visit(self, node) # test, body, orelse (ternary op)
    def visit_NamedExpr(self, node): ast.NodeVisitor.generic_visit(self, node) # Walrus operator target, value
    def visit_Starred(self, node): ast.NodeVisitor.generic_visit(self, node) # e.g. *args, **kwargs in calls or unpacking

    # --- Allowed leaf-like types (no AST children, or children are not nodes themselves) ---
    # These methods effectively say "this node type is allowed".
    def visit_Constant(self, node): pass # Python 3.8+ for num, str, bytes, bool, None
    # For older Python versions, you might need:
    # def visit_Num(self, node): pass
    # def visit_Str(self, node): pass
    # def visit_Bytes(self, node): pass
    # def visit_NameConstant(self, node): pass # True, False, None
    # def visit_Ellipsis(self, node): pass

    def visit_Name(self, node): pass # variable names, ctx (Load, Store, Del) is not an AST child
    def visit_Pass(self, node): pass
    def visit_Break(self, node): pass
    def visit_Continue(self, node): pass
    def visit_Slice(self, node): ast.NodeVisitor.generic_visit(self, node) # For subscripting, can have lower, upper, step


def safe_run_script(script_code: str):
    """
    Executes a user-provided Python script in a sandboxed environment.

    Args:
        script_code: A string containing the Python code.

    Returns:
        The result of the executed script (e.g., from a return statement or
        the value of the last expression), or None if no explicit result.

    Raises:
        ValueError: If the script involves prohibited operations,
                    contains invalid syntax, or a runtime error occurs.
    """
    try:
        # Parse the script into an AST
        tree = ast.parse(script_code)
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")

    # Validate the AST using the SafeASTVisitor
    # This will raise ValueError if any disallowed nodes or operations are found.
    validator = SafeASTVisitor()
    validator.visit(tree)

    # Prepare a safe execution environment
    # Globals dict for the executed script. Only SAFE_BUILTINS are exposed.
    safe_globals = {"__builtins__": SAFE_BUILTINS}
    
    # Locals dict will store variables created by the script.
    # We will execute the user's code wrapped in a function to capture its return value.
    user_func_name = "__safe_user_script_executor__"

    # Modify the AST: if the last statement is an expression, make it a return statement.
    # This allows scripts like "1 + 1" to return 2.
    # This is done only if there's no explicit return in the entire script.
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        has_explicit_return = any(isinstance(node, ast.Return) for node in ast.walk(tree))
        if not has_explicit_return:
            return_stmt = ast.Return(value=tree.body[-1].value)
            ast.copy_location(return_stmt, tree.body[-1])
            ast.fix_missing_locations(return_stmt)
            tree.body[-1] = return_stmt

    # Wrap the (potentially modified) user script's body in a function definition
    # def __safe_user_script_executor__():
    #   <user_script_body>
    wrapper_func_def = ast.FunctionDef(
        name=user_func_name,
        args=ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=tree.body, # Use the (potentially modified) body of the original script
        decorator_list=[],
        returns=None
    )
    ast.fix_missing_locations(wrapper_func_def)

    # Create a module containing just this function definition
    module_node = ast.Module(body=[wrapper_func_def], type_ignores=[])
    ast.fix_missing_locations(module_node)

    try:
        # Compile the module (which defines our wrapper function)
        compiled_code = compile(module_node, filename="<user_script>", mode="exec")

        # Execute the compiled code to define the wrapper function.
        # It will be defined in the `script_locals` dictionary.
        script_locals = {} 
        exec(compiled_code, safe_globals, script_locals)
        
        # Call the user's script (which is now our wrapped function)
        user_function = script_locals[user_func_name]
        result = user_function()
        return result
    except Exception as e:
        # Catch runtime errors from the user script execution
        # Re-raise as ValueError to align with the function's error handling contract.
        # Avoid leaking potentially sensitive details from arbitrary exceptions.
        raise ValueError(f"Error during script execution: {type(e).__name__}: {e}")

if __name__ == '__main__':
    # Example Usage:
    print("--- Running safe scripts ---")
    safe_scripts = [
        "1 + 2",
        "x = 10\ny = 20\nx * y",
        "print('Hello from safe script!')\n'Returned string'",
        "z = 0\nfor i in range(5):\n  z += i\nz", # Loop and assignment
        "def my_add(a, b):\n  return a + b\nmy_add(3, 4)", # Function definition
        "[i*i for i in range(5)]", # List comprehension
        "len([1,2,3])" # Calling whitelisted builtin
    ]
    for i, script in enumerate(safe_scripts):
        print(f"\nRunning script {i+1}:\n{script}")
        try:
            res = safe_run_script(script)
            print(f"Result: {res} (Type: {type(res).__name__})")
        except ValueError as e:
            print(f"Error: {e}")

    print("\n--- Running unsafe scripts (expected to fail) ---")
    unsafe_scripts = [
        "import os", # Import
        "open('file.txt', 'w')", # Disallowed function call (open is not in SAFE_BUILTINS)
        "__import__('os').system('echo unsafe')", # Attribute access / __import__
        "eval('1+1')", # Eval
        "a = {}\na.__class__", # Attribute access
        "del x", # Delete (currently disallowed)
        "1/0", # Runtime error (ZeroDivisionError)
        "while True: pass", # Infinite loop (not caught by AST, would need runtime limits)
        "lambda x: x + 1", # Lambdas are fine if their body is fine. This should pass.
                           # Let's test a bad lambda: "lambda: __import__('os')"
        "(lambda: __import__('os'))()" # Calling a lambda that does bad things
    ]
    # Adjusting the lambda test for the current attribute access restriction
    unsafe_scripts.append("(lambda: some_object.some_attribute)()")


    for i, script in enumerate(unsafe_scripts):
        print(f"\nRunning unsafe script {i+1}:\n{script}")
        try:
            res = safe_run_script(script)
            print(f"Result: {res} (Type: {type(res).__name__})")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e: # Catch any other unexpected errors for debugging
            print(f"UNEXPECTED Error: {type(e).__name__}: {e}")
    
    # Test for empty or comment-only script
    print(f"\nRunning script: '' (empty)")
    try:
        res = safe_run_script("") # ast.parse("") raises SyntaxError
        print(f"Result: {res}")
    except ValueError as e:
        print(f"Error: {e}")

    print(f"\nRunning script: '# comment only'")
    try:
        res = safe_run_script("# comment only") # ast.parse("#...") raises SyntaxError
        print(f"Result: {res}")
    except ValueError as e:
        print(f"Error: {e}")

    print(f"\nRunning script: ' ' (whitespace only)") # ast.parse(" ") is fine
    try:
        res = safe_run_script(" ")
        print(f"Result: {res}") # Should be None
    except ValueError as e:
        print(f"Error: {e}")
