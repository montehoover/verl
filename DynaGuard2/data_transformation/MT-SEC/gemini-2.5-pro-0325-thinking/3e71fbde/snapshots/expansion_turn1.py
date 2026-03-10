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
