import ast

# Define a list of disallowed module names
DISALLOWED_MODULES = [
    'os',
    'subprocess',
    'shutil',
    'sys',
    'socket',
    'ctypes',
    '_thread',
    'multiprocessing',
    'tempfile',
    'glob',
    'fnmatch',
    'platform',
    'pty',
    'signal',
    'resource',
    'fcntl',
    'termios',
    'tty',
    # 'requests', # Consider if network access is always disallowed
]

# Define a list of disallowed built-in function names (as strings)
# These are functions that, when called directly by their name, are disallowed.
DISALLOWED_BUILTIN_FUNCTIONS = [
    'open',
    'eval',
    'exec',
    'compile',
    '__import__',
    'exit',
    'quit',
    'input', # Can interact with stdin, potentially hanging or causing issues
    # 'print', # Generally safe, but can be used for DoS. Allowed for now.
]

# Define a list of disallowed attribute names
# Accessing or calling these attributes on any object is disallowed.
DISALLOWED_ATTRIBUTES = [
    # os module attributes / functions
    'system', 'popen', 'popen2', 'popen3', 'popen4', 'execl', 'execle', 'execlp',
    'execv', 'execve', 'execvp', 'fork', 'forkpty', 'kill', 'killpg', 'plock',
    'spawnl', 'spawnle', 'spawnlp', 'spawnv', 'spawnve', 'spawnvp', 'startfile',
    'remove', 'unlink', 'rmdir', 'removedirs', 'mkdir', 'makedirs', 'rename',
    'renames', 'link', 'symlink', 'chmod', 'chown', 'chroot', 'listdir', 'scandir',
    'access', 'stat', 'lstat', 'mknod', 'mkfifo', 'putenv', 'unsetenv', 'setuid',
    'setgid', 'setsid', 'setpgid', 'setpriority', 'getlogin', 'getgroups',
    'getcwdb', 'getcwd',

    # subprocess module attributes / functions
    'call', 'check_call', 'check_output', 'run', 'Popen',

    # shutil module attributes / functions
    'copy', 'copy2', 'copyfile', 'copyfileobj', 'copymode', 'copystat', 'copytree',
    'disk_usage', 'ignore_patterns', 'make_archive', 'move', 'rmtree', 'unpack_archive',
    'which',

    # socket module attributes / functions (indicative of network activity)
    # 'socket', 'create_connection', 'connect', 'bind', 'listen', 'accept', 'send', 'recv', # Covered by 'socket' in DISALLOWED_MODULES

    # Dangerous dunder attributes or attributes used for introspection/manipulation
    '__builtins__',
    '__class__',
    '__subclasses__',
    '__globals__',
    '__code__',
    '__closure__',
    '__func__',
    '__self__',
    '__mro__',
    '__bases__',
    '__dict__', # Modifying __dict__ can be dangerous
]


class CodeSafetyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.safe = True

    def visit_Import(self, node: ast.Import):
        if not self.safe: return # Stop if already found unsafe
        for alias in node.names:
            if alias.name in DISALLOWED_MODULES:
                self.safe = False
                return
        super().generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if not self.safe: return # Stop if already found unsafe
        if node.module and node.module in DISALLOWED_MODULES:
            self.safe = False
            return
        
        # Check if importing a disallowed name from an allowed/unknown module
        # e.g., from some_utility import system
        for alias in node.names:
            if alias.name in DISALLOWED_ATTRIBUTES or \
               alias.name in DISALLOWED_BUILTIN_FUNCTIONS:
                self.safe = False
                return
        super().generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if not self.safe: return # Stop if already found unsafe
        # Check for disallowed built-in function calls (e.g., eval(), open())
        if isinstance(node.func, ast.Name):
            if node.func.id in DISALLOWED_BUILTIN_FUNCTIONS:
                self.safe = False
                return
        
        # Calls to attributes (e.g., os.system()) are handled by visit_Attribute
        # because the attribute access itself (os.system) is checked.
        super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if not self.safe: return # Stop if already found unsafe
        # Check if the attribute being accessed is disallowed (e.g., .system, .__builtins__)
        if node.attr in DISALLOWED_ATTRIBUTES:
            self.safe = False
            return
        super().generic_visit(node)


def analyze_code_safety(code_string: str) -> bool:
    """
    Analyzes a string of Python code for potentially harmful operations
    by parsing it into an Abstract Syntax Tree (AST) and checking for
    disallowed modules, functions, and attributes.

    Args:
        code_string: The Python code to analyze.

    Returns:
        True if the code is deemed safe based on the defined rules,
        False otherwise (including if the code has a syntax error).
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        # Code that cannot be parsed is considered unsafe to execute.
        return False

    visitor = CodeSafetyVisitor()
    visitor.visit(tree)
    return visitor.safe

if __name__ == '__main__':
    # Example Usage and Tests:
    safe_code_examples = [
        "print('Hello, world!')",
        "a = 1 + 2\nb = a * 3\nprint(b)",
        "def my_func(x):\n  return x * x\nmy_func(5)",
        "import math\nprint(math.sqrt(16))",
        "[x for x in range(10)]",
    ]

    unsafe_code_examples = [
        ("import os", "Importing 'os' module"),
        ("import subprocess", "Importing 'subprocess' module"),
        ("from os import system", "Importing 'system' from 'os'"),
        ("from shutil import rmtree", "Importing 'rmtree' from 'shutil'"),
        ("open('some_file.txt', 'w')", "Calling 'open' function"),
        ("eval('1+1')", "Calling 'eval' function"),
        ("exec('print(\"executed\")')", "Calling 'exec' function"),
        ("os.system('ls')", "Accessing 'os.system' attribute (via ast.Attribute)"), # Caught by visit_Attribute
        ("import os\nmy_os = os\nmy_os.system('ls')", "Aliased module call to 'system'"), # Caught by import os and .system
        ("__import__('os').system('ls')", "Using __import__ to get 'os'"), # Caught by __import__ and .system
        ("x = open", "Assigning 'open' function"), # Caught by visit_Call if x() is called, or if 'open' is in DISALLOWED_ATTRIBUTES (it's not, it's a function)
                                                # This specific case (assignment only) is not caught unless 'open' is in DISALLOWED_ATTRIBUTES.
                                                # However, calling x() later would be caught if x was an alias for a disallowed function.
                                                # For now, 'open' is in DISALLOWED_BUILTIN_FUNCTIONS, so open() is caught.
        ("foo = 'os'\ngetattr(__import__(foo), 'system')('ls')", "Complex getattr with __import__"), # __import__ is caught. .system is caught.
        ("print(object.__subclasses__())", "Accessing '__subclasses__' attribute"), # Caught by visit_Attribute
        ("a = {}\na.__dict__", "Accessing '__dict__' attribute"), # Caught by visit_Attribute
        ("import sys\nsys.exit(0)", "Importing 'sys' module"),
        ("compile('print(1)', '<string>', 'exec')", "Calling 'compile' function"),
    ]

    print("Testing safe code examples:")
    for i, code in enumerate(safe_code_examples):
        is_safe = analyze_code_safety(code)
        print(f"Example {i+1}: {'SAFE' if is_safe else 'UNSAFE'} - Code: {code.splitlines()[0]}")
        assert is_safe, f"Safe code example {i+1} was incorrectly flagged as UNSAFE: {code}"

    print("\nTesting unsafe code examples:")
    for i, (code, reason) in enumerate(unsafe_code_examples):
        is_safe = analyze_code_safety(code)
        print(f"Example {i+1}: {'SAFE' if is_safe else 'UNSAFE'} - Reason: {reason} - Code: {code.splitlines()[0]}")
        assert not is_safe, f"Unsafe code example {i+1} ({reason}) was incorrectly flagged as SAFE: {code}"

    print("\nAll tests passed.")
