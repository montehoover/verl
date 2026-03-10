import ast

def analyze_ast_for_safety(code_string):
    """
    Analyzes Python code for potentially harmful operations.
    
    Args:
        code_string: String containing Python code to analyze
        
    Returns:
        bool: True if the code is safe, False otherwise
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        # Invalid Python syntax is considered unsafe
        return False
    
    # Define unsafe operations
    unsafe_nodes = (
        ast.Import,      # Importing modules could be dangerous
        ast.ImportFrom,  # Importing from modules
        ast.FunctionDef, # Defining functions with __import__ or eval
        ast.AsyncFunctionDef,
        ast.ClassDef,    # Class definitions might hide malicious code
    )
    
    unsafe_names = {
        'eval', 'exec', 'compile', '__import__', 'open',
        'file', 'input', 'raw_input', 'execfile', 'getattr',
        'setattr', 'delattr', 'globals', 'locals', 'vars',
        'dir', 'reload', 'type', 'help', 'breakpoint',
        'exit', 'quit', 'os', 'sys', 'subprocess', 'socket',
        'urllib', 'requests', 'pickle', 'shelve', 'marshal'
    }
    
    class SafetyChecker(ast.NodeVisitor):
        def __init__(self):
            self.is_safe = True
            
        def visit_Import(self, node):
            self.is_safe = False
            
        def visit_ImportFrom(self, node):
            self.is_safe = False
            
        def visit_Call(self, node):
            # Check if calling dangerous built-in functions
            if isinstance(node.func, ast.Name) and node.func.id in unsafe_names:
                self.is_safe = False
            # Check for attribute calls like os.system
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id in unsafe_names:
                    self.is_safe = False
            self.generic_visit(node)
            
        def visit_Name(self, node):
            # Check for access to dangerous names
            if node.id in unsafe_names:
                self.is_safe = False
            self.generic_visit(node)
            
        def visit_Attribute(self, node):
            # Check for dangerous attributes like __builtins__, __globals__
            dangerous_attrs = {'__builtins__', '__globals__', '__code__', 
                             '__class__', '__bases__', '__subclasses__'}
            if node.attr in dangerous_attrs:
                self.is_safe = False
            self.generic_visit(node)
    
    checker = SafetyChecker()
    checker.visit(tree)
    
    return checker.is_safe
