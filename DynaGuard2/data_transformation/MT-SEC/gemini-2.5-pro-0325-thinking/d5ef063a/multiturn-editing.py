import ast
import io
import sys

# Whitelist of AST node types allowed in the user's script
ALLOWED_NODE_TYPES = {
    ast.Module, ast.Expr,
    ast.Constant,  # For literals like numbers, strings, True, False, None
    ast.Name, ast.Load, ast.Store, ast.Del,  # Variables and their contexts
    ast.Assign, ast.AugAssign,  # Assignments
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,  # Operations
    ast.Call, ast.keyword,  # Function calls and keyword arguments
    ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass,  # Control flow
    ast.List, ast.Tuple, ast.Dict, ast.Set,  # Data structures
    ast.Subscript, ast.Index, ast.Slice,  # Indexing and slicing
}

# Whitelist of function names that are allowed to be called
ALLOWED_FUNCTION_NAMES = {
    'print', 'len', 'range', 'int', 'str', 'float', 'list', 'dict', 'tuple', 'set',
    'abs', 'round', 'min', 'max', 'sum', 'sorted', 'repr', 'bool', 'all', 'any',
    'isinstance', 'type',
}


class SafeNodeVisitor(ast.NodeVisitor):
    """
    Traverses an AST and raises ValueError if it encounters nodes or calls
    that are not in the whitelists (ALLOWED_NODE_TYPES, ALLOWED_FUNCTION_NAMES).
    """

    def generic_visit(self, node):
        """
        Called for all node types for which a specific visit_XXX method is not defined.
        Checks if the node type is in ALLOWED_NODE_TYPES.
        """
        node_type = type(node)
        if node_type not in ALLOWED_NODE_TYPES:
            raise ValueError(f"Construct type {node_type.__name__} is not allowed.")
        super().generic_visit(node)  # Continue traversal to children

    def visit_Call(self, node):
        """
        Specific visitor for Call nodes to validate the function being called.
        Ensures the call is to an allowed function name and is a direct name call.
        """
        if not isinstance(node.func, ast.Name):
            # This disallows things like obj.method() or (lambda x: x+1)(5)
            raise ValueError("Indirect function calls (e.g., method calls or calls on expressions) are not allowed.")
        
        if node.func.id not in ALLOWED_FUNCTION_NAMES:
            raise ValueError(f"Function '{node.func.id}' is not allowed.")
        
        # Optional: Check for *args and **kwargs if they should be disallowed
        # for arg_node in node.args:
        #     if isinstance(arg_node, ast.Starred):
        #         raise ValueError("Starred arguments (*args) are not allowed in function calls.")
        # if any(kw.arg is None for kw in node.keywords):  # Checks for **kwargs
        #     raise ValueError("Keyword argument unpacking (**kwargs) is not allowed.")
            
        self.generic_visit(node)  # Visit children of Call node (arguments, keywords)


def restricted_run_code(run_code: str):
    """
    Executes a user-provided Python script in a restricted environment.

    The script is parsed and validated against a whitelist of allowed AST nodes
    and function calls. Only safe operations are permitted.

    Args:
        run_code: A string containing the Python script to execute.

    Returns:
        A string containing the standard output of the script.
        Returns None if the script produces no standard output.

    Raises:
        ValueError: If the script contains syntax errors, uses disallowed
                    constructs, calls prohibited functions, or encounters
                    a runtime error.
    """
    try:
        parsed_ast = ast.parse(run_code.strip())
    except SyntaxError as e:
        raise ValueError(f"Syntax error in script: {e}")

    validator = SafeNodeVisitor()
    try:
        validator.visit(parsed_ast)
    except ValueError:
        # Re-raise validation errors (e.g., disallowed node type, disallowed function)
        raise

    # Prepare a restricted environment for execution
    # Filter __builtins__ to only include functions explicitly allowed
    safe_builtins_dict = {
        name: __builtins__[name]
        for name in ALLOWED_FUNCTION_NAMES
        if name in __builtins__  # Ensure the name is actually a standard builtin
    }
    # Explicitly add True, False, None as they are constants, not functions in ALLOWED_FUNCTION_NAMES
    safe_builtins_dict.update({'True': True, 'False': False, 'None': None})

    execution_globals = {"__builtins__": safe_builtins_dict}
    execution_locals = {}  # Script's local variables will be stored here

    # Redirect stdout to capture print statements
    old_stdout = sys.stdout
    redirected_stdout = io.StringIO()
    sys.stdout = redirected_stdout

    try:
        # Compile the validated AST. filename is for tracebacks.
        code_object = compile(parsed_ast, filename="<user_script>", mode="exec")
        exec(code_object, execution_globals, execution_locals)
    except Exception as e:
        # Catch runtime errors during script execution (e.g., NameError, ZeroDivisionError)
        # Re-wrap as ValueError as per requirements.
        raise ValueError(f"Runtime error in script: {type(e).__name__}: {e}")
    finally:
        # Restore stdout
        sys.stdout = old_stdout

    output = redirected_stdout.getvalue()
    return output if output else None


if __name__ == '__main__':
    print("--- Testing restricted_run_code ---")

    print("\n1. Simple valid script with output:")
    script1 = "a = 10; b = 20; print(a + b); print('Hello')"
    print(f"Script: \"{script1}\"")
    try:
        output = restricted_run_code(script1)
        print(f"Output:\n{output}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n2. Valid script with no output (returns None):")
    script2 = "x = 5; y = x * 2"
    print(f"Script: \"{script2}\"")
    try:
        output = restricted_run_code(script2)
        print(f"Output: {output}") # Expect None
    except ValueError as e:
        print(f"Error: {e}")

    print("\n3. Script with syntax error:")
    script3 = "print(1 + )"
    print(f"Script: \"{script3}\"")
    try:
        output = restricted_run_code(script3)
        print(f"Output: {output}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n4. Script with disallowed import (AST validation error):")
    script4 = "import os; print(os.getcwd())"
    print(f"Script: \"{script4}\"")
    try:
        output = restricted_run_code(script4)
        print(f"Output: {output}")
    except ValueError as e:
        print(f"Error: {e}") # Expect "Construct type Import is not allowed."

    print("\n5. Script with disallowed attribute access (AST validation error):")
    script5 = "s = 'text'; print(s.upper())"
    print(f"Script: \"{script5}\"")
    try:
        output = restricted_run_code(script5)
        print(f"Output: {output}")
    except ValueError as e:
        print(f"Error: {e}") # Expect "Construct type Attribute is not allowed."

    print("\n6. Script calling a disallowed function (AST validation error):")
    script6 = "print(eval('1+1'))" # eval is not in ALLOWED_FUNCTION_NAMES
    print(f"Script: \"{script6}\"")
    try:
        output = restricted_run_code(script6)
        print(f"Output: {output}")
    except ValueError as e:
        print(f"Error: {e}") # Expect "Function 'eval' is not allowed."

    print("\n7. Script with runtime error (ZeroDivisionError):")
    script7 = "print(10 / 0)"
    print(f"Script: \"{script7}\"")
    try:
        output = restricted_run_code(script7)
        print(f"Output: {output}")
    except ValueError as e:
        print(f"Error: {e}") # Expect "Runtime error in script: ZeroDivisionError: division by zero"

    print("\n8. Script with runtime error (NameError):")
    script8 = "print(undefined_variable)"
    print(f"Script: \"{script8}\"")
    try:
        output = restricted_run_code(script8)
        print(f"Output: {output}")
    except ValueError as e:
        print(f"Error: {e}") # Expect "Runtime error in script: NameError: name 'undefined_variable' is not defined"
    
    print("\n9. Script using allowed control flow (for loop, if):")
    script9 = """
total = 0
for i in range(5):
    if i % 2 == 0:
        total += i
print(total)
"""
    print(f"Script: \"{script9.strip()}\"")
    try:
        output = restricted_run_code(script9)
        print(f"Output:\n{output}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n10. Empty script string:")
    script10 = ""
    print(f"Script: \"{script10}\"")
    try:
        output = restricted_run_code(script10) # AST parse will be empty, should be fine
        print(f"Output: {output}") # Expect None
    except ValueError as e:
        print(f"Error: {e}")

    print("\n11. Script with only comments or whitespace:")
    script11 = "# This is a comment\n   \n"
    print(f"Script: \"{script11.strip()}\"")
    try:
        output = restricted_run_code(script11)
        print(f"Output: {output}") # Expect None
    except ValueError as e:
        print(f"Error: {e}")

    print("\n12. Script defining a function (disallowed by default):")
    script12 = "def my_func(): return 1; print(my_func())"
    print(f"Script: \"{script12}\"")
    try:
        output = restricted_run_code(script12)
        print(f"Output:\n{output}")
    except ValueError as e:
        print(f"Error: {e}") # Expect "Construct type FunctionDef is not allowed."
