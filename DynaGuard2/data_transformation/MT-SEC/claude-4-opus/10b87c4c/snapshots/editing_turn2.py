import ast
import sys
from io import StringIO

def execute_simple_operation(operation):
    """
    Execute a simple arithmetic operation or Python expression safely.
    
    Args:
        operation (str): A string representing a Python expression or simple code
    
    Returns:
        Any: The result of the operation or captured output
    """
    # Restrict dangerous operations
    restricted_names = {
        '__import__', 'eval', 'exec', 'compile', 'open', 'input', 
        'breakpoint', 'help', 'dir', 'vars', 'locals', 'globals'
    }
    
    restricted_modules = {
        'os', 'sys', 'subprocess', 'socket', 'requests', 'urllib',
        'importlib', 'builtins', '__builtins__'
    }
    
    # Parse the code to check for restricted operations
    try:
        tree = ast.parse(operation, mode='eval')
        # Check for simple expressions first
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in restricted_names:
                raise ValueError(f"Restricted function: {node.id}")
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                raise ValueError("Import statements not allowed")
    except SyntaxError:
        # Try parsing as statements if expression parsing fails
        try:
            tree = ast.parse(operation, mode='exec')
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in restricted_names:
                    raise ValueError(f"Restricted function: {node.id}")
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    module_name = None
                    if isinstance(node, ast.Import):
                        module_name = node.names[0].name
                    elif isinstance(node, ast.ImportFrom):
                        module_name = node.module
                    if module_name and any(module_name.startswith(restricted) for restricted in restricted_modules):
                        raise ValueError(f"Restricted module: {module_name}")
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id in restricted_modules:
                        raise ValueError(f"Restricted module: {node.value.id}")
        except Exception as e:
            raise ValueError(f"Invalid code: {e}")
    
    # Create a restricted environment
    safe_dict = {
        '__builtins__': {
            'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
            'chr': chr, 'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
            'filter': filter, 'float': float, 'format': format, 'hex': hex,
            'int': int, 'isinstance': isinstance, 'len': len, 'list': list,
            'map': map, 'max': max, 'min': min, 'oct': oct, 'ord': ord,
            'pow': pow, 'print': print, 'range': range, 'reversed': reversed,
            'round': round, 'set': set, 'sorted': sorted, 'str': str,
            'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip,
            'True': True, 'False': False, 'None': None
        }
    }
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        # Try to evaluate as expression first
        result = eval(operation, {"__builtins__": safe_dict['__builtins__']}, {})
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        if output:
            return output.strip()
        return result
    except:
        # If expression fails, try executing as statements
        try:
            exec(operation, {"__builtins__": safe_dict['__builtins__']}, {})
            sys.stdout = old_stdout
            output = captured_output.getvalue()
            return output.strip() if output else None
        except Exception as e:
            sys.stdout = old_stdout
            raise ValueError(f"Execution error: {e}")
