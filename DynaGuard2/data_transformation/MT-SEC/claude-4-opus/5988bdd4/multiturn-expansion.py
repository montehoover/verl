import ast
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

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


def execute_safe_snippet(code_string):
    """
    Execute a Python code snippet if it's deemed safe and capture any output.
    
    Args:
        code_string (str): The Python code to execute
        
    Returns:
        str: The output produced by the code
        
    Raises:
        ValueError: If the code is unsafe or invalid
    """
    # First check if the code is safe
    if not analyze_snippet_safety(code_string):
        raise ValueError("Code snippet contains unsafe operations")
    
    # Create a restricted globals environment
    safe_builtins = {
        'abs': abs,
        'all': all,
        'any': any,
        'bool': bool,
        'dict': dict,
        'enumerate': enumerate,
        'filter': filter,
        'float': float,
        'int': int,
        'len': len,
        'list': list,
        'map': map,
        'max': max,
        'min': min,
        'pow': pow,
        'range': range,
        'reversed': reversed,
        'round': round,
        'set': set,
        'slice': slice,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'zip': zip,
        'print': print,
        'True': True,
        'False': False,
        'None': None,
    }
    
    # Create a restricted environment
    safe_globals = {'__builtins__': safe_builtins}
    safe_locals = {}
    
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # Compile the code first to check for syntax errors
        compiled_code = compile(code_string, '<string>', 'exec')
        
        # Execute the code with output redirection
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compiled_code, safe_globals, safe_locals)
        
        # Combine stdout and stderr output
        output = stdout_capture.getvalue()
        error_output = stderr_capture.getvalue()
        
        if error_output:
            output += error_output
        
        return output
        
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error during execution: {str(e)}")
    finally:
        stdout_capture.close()
        stderr_capture.close()


def run_python_snippet(code_string):
    """
    Check safety and execute a Python code snippet, returning the result.
    
    Args:
        code_string (str): The Python code to execute
        
    Returns:
        str or None: The output produced by the code, or None if no output
        
    Raises:
        ValueError: If the code is unsafe or invalid
    """
    output = execute_safe_snippet(code_string)
    
    # Return None if there's no output, otherwise return the output
    return output if output else None
