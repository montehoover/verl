import ast

class SafeScriptVisitor(ast.NodeVisitor):
    """
    AST visitor to check if a script contains only allowed safe operations.
    Allowed operations include:
    - Basic arithmetic (+, -, *, /, //, %, **)
    - String literals and concatenation
    - Variable assignment and usage (names only)
    - Numeric and string constants
    - Tuple literals (if elements are safe and context is Load)

    Any other operation or AST node type will mark the script as unsafe.
    """
    def __init__(self):
        self.safe = True
        self._allowed_bin_ops = {
            ast.Add, ast.Sub, ast.Mult, ast.Div,
            ast.FloorDiv, ast.Mod, ast.Pow
        }
        self._allowed_unary_ops = {ast.UAdd, ast.USub}

    def visit(self, node):
        """Override default visit to stop early if already unsafe."""
        if not self.safe:
            return
        super().visit(node)

    def generic_visit(self, node):
        """
        Called for AST nodes for which no specific visit_NodeName method is found.
        By default, any unhandled node type is considered unsafe.
        """
        # print(f"Unsafe node type encountered by generic_visit: {type(node).__name__}") # For debugging
        self.safe = False

    # --- Whitelisted Node Type Handlers ---

    def visit_Module(self, node):
        """Root of the AST. Safe if its body is safe."""
        for stmt in node.body:
            self.visit(stmt)
            if not self.safe:  # Early exit
                return

    def visit_Expr(self, node):
        """Expression statement. Safe if its value is safe."""
        self.visit(node.value)

    def visit_Constant(self, node):
        """Safe node type (numbers, strings, None, True, False)."""
        # Check if the constant type is simple (numeric, string, boolean, None)
        if not isinstance(node.value, (int, float, complex, str, bytes, bool, type(None))):
            # Ellipsis (...) is also a constant, but might not be desired.
            # For "simple arithmetic or string manipulations", these are primary.
            self.safe = False
            # print(f"Unsafe constant type: {type(node.value)}") # For debugging
        # No children to visit.

    def visit_Name(self, node):
        """Variable names. Safe if context is Load or Store."""
        if not isinstance(node.ctx, (ast.Load, ast.Store)):
            self.safe = False
            # print(f"Unsafe context for Name: {type(node.ctx).__name__}") # For debugging

    def visit_Load(self, node):
        """Context for loading a variable. Inherently safe."""
        pass

    def visit_Store(self, node):
        """Context for storing a variable. Inherently safe."""
        pass

    def visit_Assign(self, node):
        """Assignment. Safe if targets are Names/Tuples of Names and value is safe."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.visit(target) # Checks context via visit_Name
            elif isinstance(target, ast.Tuple):
                # Ensure tuple elements are simple names for assignment
                if not all(isinstance(elt, ast.Name) and isinstance(elt.ctx, ast.Store) for elt in target.elts):
                    self.safe = False
                    # print("Unsafe element in assignment target tuple") # For debugging
                    return
                self.visit(target) # visit_Tuple will check elements and context
            else:
                self.safe = False
                # print(f"Unsafe assignment target type: {type(target).__name__}") # For debugging
                return
            if not self.safe: return

        self.visit(node.value)

    def visit_Tuple(self, node):
        """Tuple. Safe if elements are safe and context is Load or Store."""
        if not isinstance(node.ctx, (ast.Load, ast.Store)):
            self.safe = False
            # print(f"Unsafe Tuple context: {type(node.ctx).__name__}") # For debugging
            return
        for elt in node.elts:
            self.visit(elt)
            if not self.safe: return

    def visit_BinOp(self, node):
        """Binary operation. Safe if operator is whitelisted and operands are safe."""
        if type(node.op) not in self._allowed_bin_ops:
            self.safe = False
            # print(f"Disallowed BinOp operator: {type(node.op).__name__}") # For debugging
            return
        self.visit(node.left)
        if not self.safe: return
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        """Unary operation. Safe if operator is whitelisted and operand is safe."""
        if type(node.op) not in self._allowed_unary_ops:
            self.safe = False
            # print(f"Disallowed UnaryOp operator: {type(node.op).__name__}") # For debugging
            return
        self.visit(node.operand)

def filter_unsafe_operations(script_string: str) -> bool:
    """
    Checks if a Python script string contains only safe operations,
    specifically simple arithmetic or string manipulations.

    Args:
        script_string: The Python script as a string.

    Returns:
        True if the script is considered safe, False otherwise.
    """
    if not script_string.strip():
        # An empty or whitespace-only script can be considered safe.
        return True
    try:
        tree = ast.parse(script_string)
    except SyntaxError:
        # Script with syntax errors is not safe to execute.
        return False

    visitor = SafeScriptVisitor()
    visitor.visit(tree)
    return visitor.safe

def execute_safe_script(script_string: str):
    """
    Executes a Python script string that has been verified as safe by
    filter_unsafe_operations.

    Args:
        script_string: The Python script as a string.

    Returns:
        The result of the last expression in the script, or None if the
        script does not end with an expression. Returns an error message
        string if any issue occurs during execution.
    """
    if not script_string.strip():
        return None # Empty or whitespace-only script

    # Using a very restricted global scope for safety.
    # __builtins__ is empty, meaning no direct access to builtin functions.
    # filter_unsafe_operations should have already disallowed Calls to builtins.
    script_globals = {"__builtins__": {}}
    script_locals = {}

    try:
        tree = ast.parse(script_string)
        if not tree.body: # Script contains only comments
            return None

        # Compile the entire script for execution of all statements (assignments etc.)
        # Use a descriptive filename for potential tracebacks.
        code_obj = compile(tree, filename='<safe_script>', mode='exec')
        exec(code_obj, script_globals, script_locals)

        # If the last statement is an expression, evaluate it and return its result
        last_statement = tree.body[-1]
        if isinstance(last_statement, ast.Expr):
            # Compile the last expression for evaluation
            eval_expr_node = ast.Expression(body=last_statement.value)
            eval_code_obj = compile(eval_expr_node, filename='<safe_script_last_expr>', mode='eval')
            result = eval(eval_code_obj, script_globals, script_locals)
            return result
        else:
            # Script ended with an assignment or other non-expression statement
            return None
    except SyntaxError:
        # This case should ideally be caught by filter_unsafe_operations if called first.
        # Including for robustness of execute_safe_script if called directly with bad syntax.
        return "Error: Invalid syntax in script."
    except Exception as e:
        # Catch other runtime errors (NameError, ZeroDivisionError, etc.)
        return f"Error executing script: {type(e).__name__}: {e}"

if __name__ == '__main__':
    # Test cases for filter_unsafe_operations
    safe_scripts_for_filter = [
        "1 + 1",
        "x = 10\ny = 20\nz = x * y - 5 / (2 // 1) % 3 ** 2",
        "s = 'hello' + ' ' + 'world'",
        "a = -5\nb = +10", # Unary operations
        "result = (1 + 2) * 3",
        "x, y = 1, \"foo\"",
        "z = (1, 2, 'bar')", # Tuple literal
        "a = True\nb = False\nc = None", # Safe constants
        "", # Empty script
        "# This is a comment", # Comment-only script
        "x = 1 # comment after code",
        "y = x",
        "s1 = 'abc'\ns2 = 'def'\ns = s1 + s2",
        "val = 1.23 * 10e-2",
    ]

    unsafe_scripts = [
        "import os",
        "os.system('echo unsafe')",
        "print('hello')", # Call
        "eval('1+1')", # Call
        "open('file.txt', 'w')", # Call
        "def foo(): pass", # FunctionDef
        "if x > 0: result = x", # If
        "for i in range(5): result = i", # For
        "while True: break", # While
        "my_list = [1,2,3]\nx = my_list[0]", # List literal and Subscript
        "my_dict = {'a': 1}\nx = my_dict['a']", # Dict literal and Subscript
        "__import__('os').system('clear')", # Call and Attribute
        "lambda x: x + 1", # Lambda
        "a = foo.bar", # Attribute
        "del x", # Delete
        "assert True", # Assert
        "class MyClass: pass", # ClassDef
        "try:\n  x=1\nexcept:\n  pass", # Try
        "with open('f.txt') as f: pass", # With
        "x = [i for i in range(10)]", # ListComp
        "global y", # Global
        "nonlocal z", # Nonlocal
        "yield 1", # Yield
        "async def f(): await asyncio.sleep(0)", # AsyncFunctionDef, Await
        "a = 1; b = 2; c = a + b; foo(c)", # Multiple statements, one unsafe
        "x = slice(1,2)", # Call
        "x[0] = 1", # Subscript with Store context
        "x = ...", # Ellipsis constant (disallowed by current visit_Constant)
        "import sys; sys.exit(0)",
        "a = 1\nexec('print(a)')",
        "a = b'bytes string'", # bytes constant is fine
        "a = 1_000_000", # numeric literals with underscores are fine
    ]

    print("Testing safe scripts for filter_unsafe_operations:")
    for i, script in enumerate(safe_scripts_for_filter):
        is_safe = filter_unsafe_operations(script)
        script_preview = script[:30].replace('\n', '\\n') + ('...' if len(script) > 30 else '')
        print(f"Script {i+1} ('{script_preview}'): Safe? {is_safe} -> {'PASS' if is_safe else 'FAIL'}")
        assert is_safe, f"Safe script for filter failed: {script}"

    print("\nTesting unsafe scripts for filter_unsafe_operations:")
    for i, script in enumerate(unsafe_scripts):
        is_safe = filter_unsafe_operations(script)
        script_preview = script[:30].replace('\n', '\\n') + ('...' if len(script) > 30 else '')
        print(f"Script {i+1} ('{script_preview}'): Safe? {is_safe} -> {'PASS' if not is_safe else 'FAIL'}")
        assert not is_safe, f"Unsafe script for filter failed: {script}"

    # Specific test for Ellipsis constant (already part of unsafe_scripts)
    # Note: Ellipsis and complex number tests are implicitly covered by the
    # safe_scripts_for_filter and unsafe_scripts lists and their assertions.

    print("\n--- Testing execute_safe_script ---")

    scripts_to_execute = [
        ("1 + 2", 3),
        ("x = 10\nx + 5", 15),
        ("x = 10\ny = x + 5", None), # Ends with assignment
        ("s1 = 'hello'\ns2 = ' world'\ns1 + s2", "hello world"),
        ("1 / 0", "Error executing script: ZeroDivisionError: division by zero"),
        ("x = y + 1", "Error executing script: NameError: name 'y' is not defined"),
        ("", None), # Empty script
        ("# This is a comment", None), # Comment-only script
        ("x = 1\ny = 2\n(x, y, x+y)", (1,2,3)), # Returns a tuple
        ("z = (10 * 2) / 4\nz", 5.0),
        ("a = True\nb = False\na", True), # Boolean result
        ("val = 1.23e2\nval", 123.0),
        ("x = 1\n# comment\nx", 1), # Ends with expression after comment
        ("x=1\n\n\nx", 1), # Extra newlines
        ("result = -5\nresult", -5), # Test unary op result
        ("res = 2**3\nres", 8), # Test power op result
    ]

    for i, (script, expected_result) in enumerate(scripts_to_execute):
        script_preview = script[:50].replace('\n', '\\n') + ('...' if len(script) > 50 else '')
        print(f"\nExecuting script {i+1}: '{script_preview}'")
        
        is_safe_to_run = filter_unsafe_operations(script)
        print(f"Filter check: {'SAFE' if is_safe_to_run else 'UNSAFE'}")

        # All scripts in scripts_to_execute are designed to be structurally safe
        # according to filter_unsafe_operations, even if they cause runtime errors.
        if not is_safe_to_run:
             print(f"Execution Test: FAILED (script unexpectedly marked unsafe by filter)")
             assert False, f"Script '{script}' was unexpectedly marked unsafe by filter_unsafe_operations."
             continue

        actual_result = execute_safe_script(script)
        print(f"Expected: {expected_result}, Actual: {actual_result}")

        if isinstance(expected_result, str) and expected_result.startswith("Error executing script:"):
            assert isinstance(actual_result, str) and actual_result.startswith("Error executing script:"), \
                f"Script: '{script}'\nExpected error string starting with 'Error executing script:', got '{actual_result}'"
            
            # Check if the specific error type mentioned in expected_result is present in actual_result
            # e.g., from "Error executing script: ZeroDivisionError: division by zero", extract "ZeroDivisionError"
            expected_error_type = expected_result.split(":")[1].strip().split(" ")[0]
            assert expected_error_type in actual_result, \
                f"Script: '{script}'\nExpected error type '{expected_error_type}' in actual error message '{actual_result}'"
        else:
            assert actual_result == expected_result, \
                f"Script: '{script}'\nExpected: {expected_result}, Got: {actual_result}"
        
        print("Execution Test: PASS")

    # Example of an unsafe script that should not be executed
    unsafe_script_for_exec_test = "import os"
    print(f"\nAttempting to execute unsafe script: '{unsafe_script_for_exec_test}'")
    is_safe_to_run_unsafe = filter_unsafe_operations(unsafe_script_for_exec_test)
    print(f"Filter check: {'SAFE' if is_safe_to_run_unsafe else 'UNSAFE'}")
    assert not is_safe_to_run_unsafe, "Unsafe script was not caught by filter!"
    if not is_safe_to_run_unsafe:
        # Try to execute it anyway to ensure execute_safe_script would also be robust
        # (though filter_unsafe_operations should prevent this path)
        # This is more of a conceptual test, as filter_unsafe_operations should block it.
        # If filter_unsafe_operations was bypassed, execute_safe_script would still fail
        # because 'import os' is a SyntaxError in the context of mode='eval' for the last expression,
        # or would be an unhandled AST node if we didn't have filter_unsafe_operations.
        # Given filter_unsafe_operations, this path isn't naturally hit for 'import os'.
        # However, if filter_unsafe_operations had a bug and allowed it,
        # exec('import os', {'__builtins__':{}}, {}) would raise an error or do nothing harmful
        # as import is a statement, not an expression, and __builtins__ is restricted.
        print("Execution Test for unsafe script: PASS (correctly identified as unsafe by filter and not executed by main test logic)")

    # Test a script that is safe but has multiple assignments and then an expression
    multi_assign_script = "a=1\nb=2\nc=a+b\nc*2"
    print(f"\nExecuting script: '{multi_assign_script}'")
    if filter_unsafe_operations(multi_assign_script):
        res = execute_safe_script(multi_assign_script)
        print(f"Expected: 6, Actual: {res}")
        assert res == 6
        print("Execution Test: PASS")
    else:
        assert False, f"Multi-assign script '{multi_assign_script}' was unexpectedly marked unsafe by filter_unsafe_operations."
    
    print("\nAll tests completed.")
