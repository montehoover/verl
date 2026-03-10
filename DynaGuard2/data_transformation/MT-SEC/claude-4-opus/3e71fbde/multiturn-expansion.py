import ast
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

def analyze_code_safety(code_string):
    """
    Analyzes Python code for safety by checking for disallowed operations.
    
    Args:
        code_string (str): The Python code to analyze
        
    Returns:
        bool: True if the code is safe to execute, False otherwise
    """
    # Define unsafe node types and attributes
    unsafe_nodes = {
        # File I/O operations
        ast.Open,
        ast.FileInput,
        
        # Import statements (could import dangerous modules)
        ast.Import,
        ast.ImportFrom,
        
        # Dangerous built-in functions
        ast.Exec,
        ast.Eval,
    }
    
    # Define unsafe function names
    unsafe_builtins = {
        'open', 'file', 'input', 'raw_input', 'compile', 
        'exec', 'eval', 'execfile', '__import__',
        'reload', 'dir', 'globals', 'locals', 'vars',
        'getattr', 'setattr', 'delattr', 'hasattr'
    }
    
    # Define unsafe module attributes that could be used for I/O or network
    unsafe_attributes = {
        'read', 'write', 'close', 'flush', 'fileno', 'seek', 'tell',
        'readline', 'readlines', 'writelines', 'truncate',
        'send', 'recv', 'connect', 'accept', 'listen', 'bind',
        'sendto', 'recvfrom', 'sendall', 'makefile',
        'urlopen', 'urlretrieve', 'request'
    }
    
    try:
        # Parse the code string into an AST
        tree = ast.parse(code_string)
    except SyntaxError:
        # If the code has syntax errors, consider it unsafe
        return False
    
    # Walk through all nodes in the AST
    for node in ast.walk(tree):
        # Check for unsafe node types
        if type(node) in unsafe_nodes:
            return False
        
        # Check for calls to unsafe built-in functions
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in unsafe_builtins:
                return False
        
        # Check for unsafe attribute access (e.g., sys.stdout.write)
        if isinstance(node, ast.Attribute):
            if node.attr in unsafe_attributes:
                return False
        
        # Check for access to potentially dangerous modules
        if isinstance(node, ast.Name):
            if node.id in {'os', 'sys', 'subprocess', 'socket', 'urllib', 
                          'urllib2', 'httplib', 'ftplib', 'telnetlib',
                          'smtplib', 'requests', 'shutil', 'tempfile'}:
                return False
    
    # If no unsafe operations were found, the code is safe
    return True


def execute_safe_code(code_string):
    """
    Executes safe Python code and returns any output produced.
    
    Args:
        code_string (str): The Python code to execute
        
    Returns:
        str or None: The output produced by the code, or None if no output
    """
    # First check if the code is safe
    if not analyze_code_safety(code_string):
        raise ValueError("Code contains unsafe operations and cannot be executed")
    
    # Create a restricted execution environment
    safe_builtins = {
        'abs': abs,
        'all': all,
        'any': any,
        'bin': bin,
        'bool': bool,
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
        'True': True,
        'False': False,
        'None': None,
    }
    
    # Create isolated global and local namespaces
    safe_globals = {'__builtins__': safe_builtins}
    safe_locals = {}
    
    # Capture output
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    
    try:
        # Redirect stdout and stderr to capture output
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            # Execute the code in the restricted environment
            exec(code_string, safe_globals, safe_locals)
        
        # Get the captured output
        output = output_buffer.getvalue()
        error = error_buffer.getvalue()
        
        # Combine stdout and stderr
        full_output = output + error
        
        # Return output if any, otherwise None
        return full_output if full_output else None
        
    except Exception as e:
        # If an error occurs during execution, return the error message
        return f"Error during execution: {str(e)}"
    finally:
        # Clean up buffers
        output_buffer.close()
        error_buffer.close()


def run_code_snippet(code_string):
    """
    Checks code safety and executes it if safe, returning the result.
    
    Args:
        code_string (str): The Python code snippet to run
        
    Returns:
        str or None: The result of code execution, or None if no output
        
    Raises:
        ValueError: If the code contains disallowed operations
    """
    # Check if the code is safe
    if not analyze_code_safety(code_string):
        raise ValueError("Code contains disallowed operations")
    
    # Execute the safe code and return the result
    return execute_safe_code(code_string)
