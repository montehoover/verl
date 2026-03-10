import ast

def analyze_ast_for_safety(code_string):
    """
    Analyzes Python code for potentially harmful operations.
    
    Args:
        code_string (str): Python code to analyze
        
    Returns:
        bool: True if the code is safe, False otherwise
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        # Invalid syntax is considered unsafe
        return False
    
    # Define unsafe operations
    unsafe_names = {
        'eval', 'exec', 'compile', '__import__',
        'open', 'file', 'input', 'raw_input',
        'execfile', 'reload'
    }
    
    unsafe_modules = {
        'os', 'sys', 'subprocess', 'socket',
        'requests', 'urllib', 'httplib', 'ftplib',
        'telnetlib', 'ssl', 'pickle', 'shelve',
        'marshal', 'tempfile', 'glob', 'shutil',
        'importlib', 'imp', 'runpy', 'compileall'
    }
    
    class SafetyChecker(ast.NodeVisitor):
        def __init__(self):
            self.is_safe = True
            
        def visit_Import(self, node):
            for alias in node.names:
                if alias.name.split('.')[0] in unsafe_modules:
                    self.is_safe = False
            self.generic_visit(node)
            
        def visit_ImportFrom(self, node):
            if node.module and node.module.split('.')[0] in unsafe_modules:
                self.is_safe = False
            self.generic_visit(node)
            
        def visit_Call(self, node):
            # Check for direct function calls
            if isinstance(node.func, ast.Name) and node.func.id in unsafe_names:
                self.is_safe = False
            # Check for getattr calls that might access unsafe functions
            elif isinstance(node.func, ast.Name) and node.func.id == 'getattr':
                self.is_safe = False
            self.generic_visit(node)
            
        def visit_Attribute(self, node):
            # Check for dangerous attribute access like __globals__, __builtins__
            if node.attr.startswith('__') and node.attr.endswith('__'):
                dangerous_attrs = {'__globals__', '__builtins__', '__import__', 
                                 '__loader__', '__subclasses__', '__bases__',
                                 '__code__', '__class__', '__mro__'}
                if node.attr in dangerous_attrs:
                    self.is_safe = False
            self.generic_visit(node)
    
    checker = SafetyChecker()
    checker.visit(tree)
    return checker.is_safe
