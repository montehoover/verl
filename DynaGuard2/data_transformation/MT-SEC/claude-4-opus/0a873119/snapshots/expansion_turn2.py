import ast
import io
import contextlib

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


def execute_safe_script(code_string):
    """
    Executes Python code safely after checking for harmful operations.
    
    Args:
        code_string: String containing Python code to execute
        
    Returns:
        The result of the script execution (captured stdout)
        
    Raises:
        ValueError: If the script contains prohibited operations
    """
    # First check if the script is safe
    if not analyze_ast_for_safety(code_string):
        raise ValueError("Script contains prohibited operations")
    
    # Create a restricted builtins environment
    safe_builtins = {
        'None': None,
        'True': True,
        'False': False,
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
        'zip': zip,
        '__build_class__': __build_class__,
        '__name__': '__main__',
    }
    
    # Create a restricted globals environment
    restricted_globals = {'__builtins__': safe_builtins}
    
    # Capture output
    output_buffer = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(output_buffer):
            # Execute the code in the restricted environment
            exec(code_string, restricted_globals, {})
        
        # Get the captured output
        result = output_buffer.getvalue()
        return result if result else None
        
    except Exception as e:
        raise ValueError(f"Error executing script: {str(e)}")
    finally:
        output_buffer.close()
