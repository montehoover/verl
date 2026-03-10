import ast

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
