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

if __name__ == '__main__':
    # Test cases
    safe_scripts = [
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

    print("Testing safe scripts:")
    for i, script in enumerate(safe_scripts):
        is_safe = filter_unsafe_operations(script)
        print(f"Script {i+1}: {is_safe} -> {'PASS' if is_safe else 'FAIL'}")
        # assert is_safe, f"Safe script failed: {script}"

    print("\nTesting unsafe scripts:")
    for i, script in enumerate(unsafe_scripts):
        is_safe = filter_unsafe_operations(script)
        print(f"Script {i+1}: {not is_safe} -> {'PASS' if not is_safe else 'FAIL'}")
        # assert not is_safe, f"Unsafe script failed: {script}"
    
    # Specific test for Ellipsis constant
    # print("\nTesting Ellipsis constant:")
    # script_ellipsis = "x = ..."
    # is_safe_ellipsis = filter_unsafe_operations(script_ellipsis)
    # print(f"Script 'x = ...': {is_safe_ellipsis} -> {'PASS' if not is_safe_ellipsis else 'FAIL'}")
    # assert not is_safe_ellipsis, "Ellipsis constant should be unsafe by current rule"

    # Specific test for complex numbers
    # print("\nTesting complex numbers:")
    # script_complex = "x = 1 + 2j\ny = complex(1,2)"
    # is_safe_complex_literal = filter_unsafe_operations("x = 1 + 2j")
    # print(f"Script 'x = 1 + 2j': {is_safe_complex_literal} -> {'PASS' if is_safe_complex_literal else 'FAIL'}")
    # assert is_safe_complex_literal

    # is_safe_complex_call = filter_unsafe_operations("y = complex(1,2)") # complex() is a Call
    # print(f"Script 'y = complex(1,2)': {is_safe_complex_call} -> {'PASS' if not is_safe_complex_call else 'FAIL'}")
    # assert not is_safe_complex_call
