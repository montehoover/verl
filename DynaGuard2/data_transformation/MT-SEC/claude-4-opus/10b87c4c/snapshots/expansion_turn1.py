import ast

def analyze_code_safety(code: str) -> bool:
    """
    Analyze Python code for safety by checking for disallowed operations.
    
    Args:
        code: String of Python code to analyze
        
    Returns:
        bool: True if code is safe to execute, False otherwise
    """
    # Define disallowed AST node types and attributes
    disallowed_nodes = {
        # File I/O operations
        ast.Open,
        
        # Import statements (could import dangerous modules)
        ast.Import,
        ast.ImportFrom,
        
        # Eval/exec operations
        ast.Eval,
        ast.Exec,
        
        # System operations
        ast.Global,
        ast.Nonlocal,
    }
    
    # Disallowed built-in functions and modules
    disallowed_names = {
        # File operations
        'open', 'file', 'input', 'raw_input',
        
        # Code execution
        'eval', 'exec', 'compile', '__import__',
        
        # System operations
        'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr', 'hasattr',
        
        # Process/system operations
        'exit', 'quit',
    }
    
    # Disallowed attribute access patterns
    disallowed_attributes = {
        '__builtins__', '__globals__', '__code__',
        '__class__', '__bases__', '__subclasses__',
        '__import__', '__loader__', '__module__',
    }
    
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
    except SyntaxError:
        # Invalid syntax is considered unsafe
        return False
    
    # Walk through all nodes in the AST
    for node in ast.walk(tree):
        # Check for disallowed node types
        if type(node) in disallowed_nodes:
            return False
        
        # Check for calls to disallowed functions
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in disallowed_names:
                return False
            
            # Check for calls like getattr(), setattr() etc.
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in disallowed_names:
                    return False
        
        # Check for name references to disallowed functions
        if isinstance(node, ast.Name) and node.id in disallowed_names:
            return False
        
        # Check for attribute access to disallowed attributes
        if isinstance(node, ast.Attribute):
            if node.attr in disallowed_attributes:
                return False
        
        # Check for string attribute access (e.g., getattr with string)
        if isinstance(node, ast.Str):
            if node.s in disallowed_attributes:
                return False
        
        # For Python 3.8+, check ast.Constant nodes as well
        if hasattr(ast, 'Constant') and isinstance(node, ast.Constant):
            if isinstance(node.value, str) and node.value in disallowed_attributes:
                return False
    
    # If no disallowed operations found, code is safe
    return True


# Example usage and tests
if __name__ == "__main__":
    # Safe code examples
    safe_examples = [
        "x = 1 + 2",
        "def add(a, b): return a + b",
        "result = [i**2 for i in range(10)]",
        "print('Hello, World!')",
    ]
    
    # Unsafe code examples
    unsafe_examples = [
        "import os",
        "open('file.txt', 'r')",
        "eval('1 + 1')",
        "__import__('os')",
        "globals()['x'] = 1",
    ]
    
    print("Safe code examples:")
    for code in safe_examples:
        print(f"  {repr(code)}: {analyze_code_safety(code)}")
    
    print("\nUnsafe code examples:")
    for code in unsafe_examples:
        print(f"  {repr(code)}: {analyze_code_safety(code)}")
