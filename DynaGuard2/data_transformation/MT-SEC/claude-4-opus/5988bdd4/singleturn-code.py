import ast

def run_python_snippet(snippet_string: str):
    """
    Executes a Python code snippet safely, ensuring only benign actions are carried out.
    
    Args:
        snippet_string: str - The user-provided Python code to be executed
        
    Returns:
        The resulting value if the code produces any output, or None otherwise
        
    Raises:
        ValueError: If the snippet attempts forbidden operations or contains invalid code
    """
    # Define forbidden names and attributes that could be dangerous
    FORBIDDEN_NAMES = {
        '__import__', 'eval', 'exec', 'compile', 'open', 'input',
        'file', 'execfile', 'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr', 'hasattr', '__builtins__'
    }
    
    FORBIDDEN_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'socket', 'urllib',
        'requests', 'pathlib', 'importlib', 'pickle', 'shelve',
        'tempfile', 'glob', 'io', 'webbrowser', 'platform'
    }
    
    FORBIDDEN_ATTRIBUTES = {
        '__class__', '__bases__', '__subclasses__', '__globals__',
        '__code__', '__closure__', '__func__', '__self__', '__module__',
        '__dict__', '__weakref__', '__qualname__', '__annotations__',
        '__mro__', '__init__', '__new__', '__del__', '__getattr__',
        '__setattr__', '__delattr__', '__getattribute__', '__call__'
    }
    
    class SafetyValidator(ast.NodeVisitor):
        """AST visitor to check for potentially dangerous operations"""
        
        def __init__(self):
            self.errors = []
            
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in FORBIDDEN_NAMES:
                self.errors.append(f"Forbidden name: {node.id}")
            self.generic_visit(node)
            
        def visit_Import(self, node):
            for alias in node.names:
                if alias.name.split('.')[0] in FORBIDDEN_MODULES:
                    self.errors.append(f"Forbidden import: {alias.name}")
            self.generic_visit(node)
            
        def visit_ImportFrom(self, node):
            if node.module and node.module.split('.')[0] in FORBIDDEN_MODULES:
                self.errors.append(f"Forbidden import: {node.module}")
            self.generic_visit(node)
            
        def visit_Attribute(self, node):
            if node.attr in FORBIDDEN_ATTRIBUTES:
                self.errors.append(f"Forbidden attribute: {node.attr}")
            self.generic_visit(node)
            
        def visit_Call(self, node):
            # Check for dangerous built-in functions
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
                self.errors.append(f"Forbidden function call: {node.func.id}")
            self.generic_visit(node)
    
    try:
        # Parse the code into an AST
        tree = ast.parse(snippet_string, mode='exec')
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}")
    
    # Validate the AST for safety
    validator = SafetyValidator()
    validator.visit(tree)
    
    if validator.errors:
        raise ValueError(f"Security violation detected: {'; '.join(validator.errors)}")
    
    # Create a restricted environment for execution
    safe_globals = {
        '__builtins__': {
            'print': print,
            'len': len,
            'range': range,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'any': any,
            'all': all,
            'True': True,
            'False': False,
            'None': None,
        }
    }
    
    safe_locals = {}
    
    try:
        # Compile and execute the code
        compiled_code = compile(tree, '<snippet>', 'exec')
        exec(compiled_code, safe_globals, safe_locals)
        
        # Check if the last statement is an expression that returns a value
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            # Re-evaluate the last expression to get its value
            last_expr = ast.Expression(body=tree.body[-1].value)
            compiled_expr = compile(last_expr, '<snippet>', 'eval')
            return eval(compiled_expr, safe_globals, safe_locals)
        
        return None
        
    except Exception as e:
        raise ValueError(f"Error executing code: {e}")
