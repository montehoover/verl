import ast

def analyze_code_safety(code_string):
    """
    Analyze Python code for potentially harmful operations.
    
    Args:
        code_string: String containing Python code to analyze
        
    Returns:
        bool: True if code is safe to execute, False otherwise
    """
    # List of unsafe built-in functions and modules
    unsafe_builtins = {
        'exec', 'eval', 'compile', '__import__', 'open', 'file',
        'input', 'raw_input', 'execfile', 'reload', 'globals', 'locals',
        'vars', 'dir', 'getattr', 'setattr', 'delattr', 'hasattr'
    }
    
    unsafe_modules = {
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'urllib2',
        'httplib', 'ftplib', 'telnetlib', 'ssl', 'pickle', 'shelve',
        'marshal', 'tempfile', 'fileinput', 'io', 'popen2', 'commands',
        'multiprocessing', 'threading', 'thread', 'signal', 'resource',
        'gc', 'ctypes', 'platform', 'importlib', 'imp', 'code',
        'codeop', 'runpy', 'builtins', '__builtin__', 'webbrowser',
        'antigravity', 'this', 'shutil', 'glob', 'pathlib'
    }
    
    try:
        # Parse the code into an AST
        tree = ast.parse(code_string)
    except SyntaxError:
        # Syntax errors make code non-executable, so it's "safe" in that sense
        return False
    
    # Check for dangerous AST nodes
    for node in ast.walk(tree):
        # Check for import statements
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in unsafe_modules:
                        return False
            else:
                if node.module and node.module.split('.')[0] in unsafe_modules:
                    return False
        
        # Check for function calls
        elif isinstance(node, ast.Call):
            # Check if calling a name directly
            if isinstance(node.func, ast.Name):
                if node.func.id in unsafe_builtins:
                    return False
            # Check for attribute access (e.g., os.system)
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id in unsafe_modules:
                        return False
        
        # Check for exec/eval even in string form
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
            if 'exec' in node.value.s or 'eval' in node.value.s:
                return False
        
        # Check for file operations through context managers
        elif isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.context_expr, ast.Call):
                    if isinstance(item.context_expr.func, ast.Name):
                        if item.context_expr.func.id == 'open':
                            return False
        
        # Check for class definitions that might override builtins
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in {'type', 'object'}:
                    return False
        
        # Check for function definitions with dangerous decorators
        elif isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    if decorator.id in {'property', 'staticmethod', 'classmethod'}:
                        # These are generally safe
                        continue
                return False
        
        # Check for attribute access that might be dangerous
        elif isinstance(node, ast.Attribute):
            # Block access to __class__, __bases__, __subclasses__, etc.
            if node.attr.startswith('__') and node.attr.endswith('__'):
                dangerous_attrs = {
                    '__class__', '__bases__', '__subclasses__', '__init__',
                    '__globals__', '__code__', '__closure__', '__module__',
                    '__dict__', '__func__', '__self__', '__annotations__',
                    '__mro__', '__name__', '__qualname__', '__builtins__'
                }
                if node.attr in dangerous_attrs:
                    return False
        
        # Check for list comprehensions with suspicious operations
        elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            # These are checked recursively through ast.walk
            pass
    
    # If no dangerous operations found, code is safe
    return True
