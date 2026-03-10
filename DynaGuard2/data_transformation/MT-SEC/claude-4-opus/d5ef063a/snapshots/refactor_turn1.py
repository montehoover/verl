import ast

def restricted_run_code(run_code):
    # Define allowed names and functions
    allowed_builtins = {
        'print', 'len', 'range', 'int', 'float', 'str', 'bool',
        'list', 'dict', 'tuple', 'set', 'abs', 'min', 'max',
        'sum', 'round', 'sorted', 'enumerate', 'zip', 'map',
        'filter', 'any', 'all', 'True', 'False', 'None'
    }
    
    # Parse the code to check for safety
    try:
        tree = ast.parse(run_code)
    except SyntaxError:
        raise ValueError("Syntax error in the provided code")
    
    # Check for prohibited operations
    class SafetyChecker(ast.NodeVisitor):
        def __init__(self):
            self.safe = True
            self.error_msg = None
            
        def visit_Import(self, node):
            self.safe = False
            self.error_msg = "Import statements are not allowed"
            
        def visit_ImportFrom(self, node):
            self.safe = False
            self.error_msg = "Import statements are not allowed"
            
        def visit_FunctionDef(self, node):
            # Check for __import__ or other dangerous function names
            if node.name.startswith('_'):
                self.safe = False
                self.error_msg = "Functions starting with underscore are not allowed"
            self.generic_visit(node)
            
        def visit_Attribute(self, node):
            # Prevent access to dangerous attributes
            if isinstance(node.attr, str) and node.attr.startswith('_'):
                self.safe = False
                self.error_msg = "Access to private attributes is not allowed"
            self.generic_visit(node)
            
        def visit_Name(self, node):
            # Check for dangerous built-in names
            if node.id in ['__import__', 'eval', 'exec', 'compile', 'open', 
                          'file', 'input', 'raw_input', '__builtins__',
                          'globals', 'locals', 'vars', 'dir', 'getattr',
                          'setattr', 'delattr', 'hasattr', '__name__',
                          '__file__', '__dict__']:
                self.safe = False
                self.error_msg = f"Use of '{node.id}' is not allowed"
            self.generic_visit(node)
    
    checker = SafetyChecker()
    checker.visit(tree)
    
    if not checker.safe:
        raise ValueError(checker.error_msg)
    
    # Create a restricted environment
    safe_globals = {
        '__builtins__': {name: __builtins__[name] for name in allowed_builtins if name in __builtins__}
    }
    safe_locals = {}
    
    # Capture output
    output_lines = []
    original_print = print
    
    def capture_print(*args, **kwargs):
        output_lines.append(' '.join(str(arg) for arg in args))
    
    # Replace print in the safe environment
    if 'print' in safe_globals['__builtins__']:
        safe_globals['__builtins__']['print'] = capture_print
    
    try:
        exec(run_code, safe_globals, safe_locals)
        return '\n'.join(output_lines) if output_lines else None
    except Exception as e:
        raise ValueError(f"Error during execution: {str(e)}")
