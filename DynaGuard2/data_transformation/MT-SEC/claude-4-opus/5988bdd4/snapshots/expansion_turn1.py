import ast

def analyze_snippet_safety(code_string):
    """
    Analyze a Python code snippet for potentially harmful operations.
    
    Args:
        code_string (str): The Python code to analyze
        
    Returns:
        bool: True if the code is safe, False if potentially harmful operations are found
    """
    # List of unsafe AST node types and unsafe built-in functions
    unsafe_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
        ast.Exec,  # For Python 2 compatibility
    )
    
    unsafe_builtins = {
        'open', 'file', 'input', 'raw_input', 'eval', 'exec', 'compile',
        '__import__', 'execfile', 'reload', 'dir', 'globals', 'locals',
        'vars', 'getattr', 'setattr', 'delattr', 'hasattr',
        'type', 'isinstance', 'issubclass', 'super', 'property',
        'staticmethod', 'classmethod', 'callable', 'format',
        'memoryview', 'bytearray', 'bytes', 'chr', 'ord',
        'bin', 'hex', 'oct', 'id', 'object', 'ascii', 'repr',
        'breakpoint', 'help', 'print',  # print could be used for information disclosure
    }
    
    unsafe_attributes = {
        '__dict__', '__class__', '__bases__', '__name__', '__mro__',
        '__subclasses__', '__init__', '__globals__', '__code__',
        '__closure__', '__annotations__', '__kwdefaults__',
        '__builtins__', '__import__', '__loader__', '__package__',
        '__spec__', '__path__', '__file__', '__cached__',
    }
    
    try:
        tree = ast.parse(code_string, mode='exec')
    except SyntaxError:
        # Invalid syntax is considered unsafe
        return False
    
    class SafetyChecker(ast.NodeVisitor):
        def __init__(self):
            self.is_safe = True
            
        def visit_Import(self, node):
            self.is_safe = False
            
        def visit_ImportFrom(self, node):
            self.is_safe = False
            
        def visit_FunctionDef(self, node):
            self.is_safe = False
            
        def visit_AsyncFunctionDef(self, node):
            self.is_safe = False
            
        def visit_ClassDef(self, node):
            self.is_safe = False
            
        def visit_Delete(self, node):
            self.is_safe = False
            
        def visit_Global(self, node):
            self.is_safe = False
            
        def visit_Nonlocal(self, node):
            self.is_safe = False
            
        def visit_Call(self, node):
            # Check if calling unsafe built-in functions
            if isinstance(node.func, ast.Name) and node.func.id in unsafe_builtins:
                self.is_safe = False
            # Check for calls like getattr(), setattr(), etc.
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in unsafe_builtins:
                    self.is_safe = False
            self.generic_visit(node)
            
        def visit_Attribute(self, node):
            # Check for accessing unsafe attributes
            if node.attr in unsafe_attributes:
                self.is_safe = False
            self.generic_visit(node)
            
        def visit_Name(self, node):
            # Check if using unsafe built-in names
            if isinstance(node.ctx, ast.Load) and node.id in unsafe_builtins:
                self.is_safe = False
            self.generic_visit(node)
            
        def visit_With(self, node):
            # With statements often involve file operations
            self.is_safe = False
            
        def visit_AsyncWith(self, node):
            self.is_safe = False
            
        def visit_Try(self, node):
            # Try blocks might be attempting to catch security exceptions
            self.is_safe = False
            
        def visit_Raise(self, node):
            # Raising exceptions could be used maliciously
            self.is_safe = False
            
        def visit_Assert(self, node):
            # Assertions might be used to probe the system
            self.is_safe = False
            
        def visit_Exec(self, node):
            # For Python 2 compatibility
            self.is_safe = False
            
        def visit_Lambda(self, node):
            # Lambda functions can hide malicious code
            self.is_safe = False
            
        def visit_Yield(self, node):
            # Generators might be used for complex attacks
            self.is_safe = False
            
        def visit_YieldFrom(self, node):
            self.is_safe = False
    
    checker = SafetyChecker()
    checker.visit(tree)
    
    return checker.is_safe
