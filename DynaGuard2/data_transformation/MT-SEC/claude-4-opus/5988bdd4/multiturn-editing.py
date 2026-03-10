import sys
from io import StringIO
import ast

def run_python_snippet(snippet_string):
    """Execute a string of Python code safely and return the output."""
    # Parse the code to check for syntax errors
    try:
        tree = ast.parse(snippet_string)
    except SyntaxError:
        raise ValueError("Invalid Python syntax")
    
    # Check for potentially dangerous operations
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(alias.name in ['os', 'sys', 'subprocess', '__builtins__', 'open', 'file', 'input', 'raw_input', 'execfile', 'compile'] 
                   for alias in node.names):
                raise ValueError("Forbidden import detected")
        elif isinstance(node, ast.Name):
            if node.id in ['__import__', 'eval', 'exec', 'open', 'file', 'input', 'raw_input', 'execfile', 'compile']:
                raise ValueError("Forbidden operation detected")
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Execute the code in a restricted namespace
        restricted_globals = {
            '__builtins__': {
                'print': print,
                'range': range,
                'len': len,
                'int': int,
                'float': float,
                'str': str,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'bool': bool,
                'True': True,
                'False': False,
                'None': None,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'any': any,
                'all': all,
                'round': round,
                'pow': pow,
                'divmod': divmod,
                'isinstance': isinstance,
                'issubclass': issubclass,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'delattr': delattr,
                'callable': callable,
                'type': type,
                'dir': dir,
                'id': id,
                'ord': ord,
                'chr': chr,
                'bin': bin,
                'hex': hex,
                'oct': oct,
            }
        }
        
        exec(snippet_string, restricted_globals)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        return output.rstrip() if output else None
        
    except Exception as e:
        sys.stdout = old_stdout
        raise ValueError(f"Execution error: {str(e)}")
