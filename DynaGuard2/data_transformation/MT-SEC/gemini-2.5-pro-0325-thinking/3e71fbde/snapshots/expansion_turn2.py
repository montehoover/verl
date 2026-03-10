import ast

class CodeSafetyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.safe = True
        # Common built-in functions that are often restricted
        self.DISALLOWED_BUILTIN_FUNCTIONS = {
            'open',       # File I/O
            'eval',       # Execute arbitrary code
            'exec',       # Execute arbitrary code
        }
        # Modules that provide potentially unsafe operations
        self.DISALLOWED_MODULES = {
            'os',         # Operating system interface
            'sys',        # System-specific parameters and functions
            'subprocess', # Subprocess management
            'shutil',     # High-level file operations
            'socket',     # Network connections
            'ftplib',     # FTP protocol client
            'http',       # HTTP protocol (client and server)
            'urllib',     # URL handling module
            'requests',   # HTTP library
            'ctypes',     # Foreign function library
            'cgi',        # Common Gateway Interface support
            'pickle',     # Object serialization (potential for code execution)
            'marshal',    # Internal Python object serialization (similar risks)
        }

    def visit(self, node):
        """Override visit to stop early if already unsafe."""
        if not self.safe:
            return
        super().visit(node)

    def generic_visit(self, node):
        """Override generic_visit to stop early if already unsafe before visiting children."""
        if not self.safe:
            return
        super().generic_visit(node)

    def visit_Call(self, node):
        # Check for disallowed built-in function calls
        if isinstance(node.func, ast.Name):
            if node.func.id in self.DISALLOWED_BUILTIN_FUNCTIONS:
                self.safe = False
            elif node.func.id == '__import__': # Direct call to __import__
                self.safe = False
        
        if not self.safe: # If this node's check made it unsafe, stop.
            return
        super().generic_visit(node) # Otherwise, continue to visit children of this Call node.

    def visit_Import(self, node):
        for alias in node.names:
            # alias.name is the original module name (e.g., "os" or "os.path")
            module_base_name = alias.name.split('.')[0]
            if module_base_name in self.DISALLOWED_MODULES:
                self.safe = False
                break # Found a disallowed import
        
        if not self.safe:
            return
        super().generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            # node.module is the name of the module from which imports are made (e.g., "os" or "os.path")
            module_base_name = node.module.split('.')[0]
            if module_base_name in self.DISALLOWED_MODULES:
                self.safe = False
        
        if not self.safe:
            return
        super().generic_visit(node)

def analyze_code_safety(code_string: str) -> bool:
    """
    Analyzes a string of Python code for disallowed operations using AST.

    Checks for:
    - Use of disallowed built-in functions (e.g., open, eval, exec).
    - Importing disallowed modules (e.g., os, sys, subprocess, socket).
    - Direct calls to __import__.

    Args:
        code_string: The Python code to analyze.

    Returns:
        True if the code is considered safe based on the defined rules, False otherwise.
        Returns False if the code cannot be parsed due to SyntaxError.
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return False  # Code that doesn't parse is considered unsafe

    visitor = CodeSafetyVisitor()
    visitor.visit(tree)
    return visitor.safe

import io
import contextlib

def execute_safe_code(code_string: str):
    """
    Executes a string of Python code in an isolated environment and captures its output.

    Args:
        code_string: The Python code to execute. Assumed to have been vetted
                     by analyze_code_safety.

    Returns:
        The output (stdout) produced by the code as a string, or None if no output.
        Returns an error message string if an exception occurs during execution.
    """
    # Create a restricted environment for execution
    # Allow access to a limited set of built-ins for safety.
    # __builtins__ can be a dictionary. If you provide your own __builtins__ dictionary,
    # Python will not automatically include the standard built-in functions.
    # For now, we'll provide a very minimal set.
    # A more robust solution might involve a more carefully curated list of allowed builtins.
    allowed_builtins = {
        'print': print,
        'range': range,
        'len': len,
        'str': str,
        'int': int,
        'float': float,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'True': True,
        'False': False,
        'None': None,
        # Add other safe built-ins as needed
    }
    restricted_globals = {"__builtins__": allowed_builtins}
    restricted_locals = {}

    # Capture stdout
    stdout_capture = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(stdout_capture):
            # Execute the code in the restricted environment
            # Note: exec() itself can be dangerous if the code_string is not properly vetted.
            # The analyze_code_safety function is crucial here.
            exec(code_string, restricted_globals, restricted_locals)
        
        output = stdout_capture.getvalue()
        return output if output else None
    except Exception as e:
        # In a real application, you might want to log this error
        # or handle it more gracefully.
        return f"Error during execution: {e}"
    finally:
        stdout_capture.close()
