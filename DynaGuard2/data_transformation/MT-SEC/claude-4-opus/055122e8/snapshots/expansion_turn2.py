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


def safe_execute(code_string):
    """
    Executes Python code only if it's deemed safe by analyze_ast_for_safety.
    
    Args:
        code_string (str): Python code to execute
        
    Returns:
        Any: The result of the script if there is one, or None
        
    Raises:
        ValueError: If the script is unsafe or contains invalid syntax
    """
    try:
        # First parse to check for syntax errors
        ast.parse(code_string)
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")
    
    # Check if code is safe
    if not analyze_ast_for_safety(code_string):
        raise ValueError("Script contains unsafe operations")
    
    # Create restricted globals
    safe_globals = {
        '__builtins__': {
            'abs': abs,
            'all': all,
            'any': any,
            'ascii': ascii,
            'bin': bin,
            'bool': bool,
            'bytearray': bytearray,
            'bytes': bytes,
            'chr': chr,
            'complex': complex,
            'dict': dict,
            'divmod': divmod,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'format': format,
            'frozenset': frozenset,
            'hex': hex,
            'int': int,
            'isinstance': isinstance,
            'issubclass': issubclass,
            'iter': iter,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'next': next,
            'oct': oct,
            'ord': ord,
            'pow': pow,
            'print': print,
            'range': range,
            'repr': repr,
            'reversed': reversed,
            'round': round,
            'set': set,
            'slice': slice,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,
            'None': None,
            'True': True,
            'False': False,
        }
    }
    
    # Execute the code
    result = None
    exec(code_string, safe_globals)
    
    # Check if there's a result variable in the namespace
    if 'result' in safe_globals:
        result = safe_globals['result']
    
    return result
