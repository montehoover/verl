import ast

# Define forbidden names and operations as module constants
FORBIDDEN_NAMES = {
    '__import__', 'eval', 'exec', 'compile', 'open', 'input', 'file',
    'execfile', 'reload', 'globals', 'locals', 'vars', 'dir',
    'getattr', 'setattr', 'delattr', 'hasattr', 'type', 'isinstance',
    'issubclass', 'callable', 'classmethod', 'staticmethod', 'property',
    'super', 'object', '__builtins__', '__loader__', '__name__',
    '__package__', '__spec__', '__file__', '__cached__', '__doc__'
}

FORBIDDEN_MODULES = {
    'os', 'sys', 'subprocess', 'socket', 'requests', 'urllib',
    'shutil', 'tempfile', 'glob', 'pickle', 'marshal', 'importlib',
    'imp', 'runpy', 'pkgutil', 'inspect', 'ast', 'dis', 'code',
    'codeop', 'compileall', 'py_compile', 'builtins', '__builtin__',
    'io', 'pathlib', 'fileinput', 'linecache', 'stat', 'filecmp',
    'tempfile', 'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma',
    'csv', 'configparser', 'netrc', 'xdrlib', 'plistlib',
    'platform', 'errno', 'ctypes', 'fcntl', 'mmap', 'readline',
    'rlcompleter', 'pty', 'tty', 'termios', 'crypt', 'grp',
    'pwd', 'spwd', 'nis', 'syslog', 'commands', 'popen2',
    'webbrowser', 'cgi', 'cgitb', 'wsgiref', 'urllib', 'http',
    'ftplib', 'poplib', 'imaplib', 'nntplib', 'smtplib', 'smtpd',
    'telnetlib', 'uuid', 'socketserver', 'xmlrpc', 'ipaddress',
    'ssl', 'email', 'json', 'mailcap', 'mailbox', 'mimetypes',
    'base64', 'binhex', 'binascii', 'quopri', 'uu', 'html',
    'xml', 'dbm', 'sqlite3', 'zlib', 'gzip', 'bz2', 'lzma',
    'zipimport', 'pkgutil', 'modulefinder', 'runpy', 'trace',
    'traceback', 'pdb', 'bdb', 'cmd', 'pstats', 'timeit',
    'test', 'doctest', 'unittest', '2to3', 'distutils',
    'ensurepip', 'venv', 'asyncio', 'asyncore', 'asynchat',
    'concurrent', 'multiprocessing', 'threading', 'queue',
    'select', 'selectors', 'signal', 'mmap', 'codecs',
    'encodings', 'reprlib', 'atexit', 'gc', 'contextlib'
}

SAFE_BUILTINS = {
    'None': None,
    'True': True,
    'False': False,
    'int': int,
    'float': float,
    'str': str,
    'bool': bool,
    'list': list,
    'dict': dict,
    'tuple': tuple,
    'set': set,
    'frozenset': frozenset,
    'len': len,
    'range': range,
    'enumerate': enumerate,
    'zip': zip,
    'map': map,
    'filter': filter,
    'sum': sum,
    'min': min,
    'max': max,
    'abs': abs,
    'round': round,
    'sorted': sorted,
    'reversed': reversed,
    'all': all,
    'any': any,
    'print': print,
    'chr': chr,
    'ord': ord,
    'bin': bin,
    'hex': hex,
    'oct': oct,
    'pow': pow,
    'divmod': divmod,
    'isinstance': isinstance,
    'issubclass': issubclass,
    'slice': slice,
    'complex': complex,
    'bytes': bytes,
    'bytearray': bytearray,
    'memoryview': memoryview,
    'type': type,
    'repr': repr,
    'hash': hash,
    'id': id,
    'help': help,
    'iter': iter,
    'next': next,
    'format': format,
    'ascii': ascii,
}


class SafetyValidator(ast.NodeVisitor):
    def __init__(self):
        self.is_safe = True
        self.error_msg = None
        
    def visit_Import(self, node):
        for alias in node.names:
            if alias.name in FORBIDDEN_MODULES:
                self.is_safe = False
                self.error_msg = f"Import of module '{alias.name}' is not allowed"
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module in FORBIDDEN_MODULES:
            self.is_safe = False
            self.error_msg = f"Import from module '{node.module}' is not allowed"
        self.generic_visit(node)
        
    def visit_Name(self, node):
        if node.id in FORBIDDEN_NAMES:
            self.is_safe = False
            self.error_msg = f"Use of '{node.id}' is not allowed"
        self.generic_visit(node)
        
    def visit_Attribute(self, node):
        # Check for dangerous attribute access patterns
        if isinstance(node.value, ast.Name):
            if node.value.id == '__builtins__':
                self.is_safe = False
                self.error_msg = "Access to __builtins__ is not allowed"
            elif node.attr.startswith('_'):
                self.is_safe = False
                self.error_msg = f"Access to private attribute '{node.attr}' is not allowed"
        self.generic_visit(node)
        
    def visit_Call(self, node):
        # Check for dangerous function calls
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
            self.is_safe = False
            self.error_msg = f"Call to '{node.func.id}' is not allowed"
        self.generic_visit(node)


def parse_and_validate_code(snippet_code: str) -> ast.AST:
    """
    Parse Python code string into AST and validate it for safety.
    
    Args:
        snippet_code: The Python code to parse and validate
        
    Returns:
        The parsed AST tree if valid and safe
        
    Raises:
        ValueError: If the code has syntax errors or contains forbidden operations
    """
    try:
        # Parse the code to AST
        tree = ast.parse(snippet_code, mode='exec')
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}")
    
    # Validate the AST for safety
    validator = SafetyValidator()
    validator.visit(tree)
    
    if not validator.is_safe:
        raise ValueError(f"Forbidden operation detected: {validator.error_msg}")
    
    return tree


def create_safe_execution_environment():
    """
    Create a restricted global and local environment for code execution.
    
    Returns:
        Tuple of (safe_globals, safe_locals) dictionaries
    """
    safe_globals = {
        '__builtins__': SAFE_BUILTINS
    }
    safe_locals = {}
    
    return safe_globals, safe_locals


def execute_ast_in_sandbox(tree: ast.AST, safe_globals: dict, safe_locals: dict):
    """
    Execute an AST tree in a sandboxed environment.
    
    Args:
        tree: The AST tree to execute
        safe_globals: The restricted global environment
        safe_locals: The local environment for execution
        
    Returns:
        The result of the execution if any, None otherwise
        
    Raises:
        ValueError: If an error occurs during execution
    """
    try:
        # Compile and execute the code
        compiled = compile(tree, '<string>', 'exec')
        
        # Execute the compiled code
        exec(compiled, safe_globals, safe_locals)
        
        # If the code assigned a result to a variable, return the last expression
        if safe_locals:
            # Get the last assigned value
            result = list(safe_locals.values())[-1]
            return result
            
        return None
        
    except Exception as e:
        raise ValueError(f"Error executing code: {e}")


def execute_python_code(snippet_code: str):
    """
    Executes a string of Python code and ensures that only benign actions are carried out.
    
    Args:
        snippet_code: str, the user-provided Python code to be executed
        
    Returns:
        Returns the resulting value if the code produces any output, or None otherwise
        
    Raises:
        ValueError: if the snippet attempts forbidden operations or contains invalid code
    """
    # Parse and validate the code
    tree = parse_and_validate_code(snippet_code)
    
    # Create safe execution environment
    safe_globals, safe_locals = create_safe_execution_environment()
    
    # Execute the validated code in sandbox
    result = execute_ast_in_sandbox(tree, safe_globals, safe_locals)
    
    return result
