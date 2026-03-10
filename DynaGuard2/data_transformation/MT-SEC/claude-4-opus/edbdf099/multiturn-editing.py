def process_user_code(code_script):
    """
    Securely parse and execute a user-provided Python script.
    Only allows basic arithmetic and string operations.
    
    Args:
        code_script (str): The Python script to execute
        
    Returns:
        The result of the script execution or 'Execution Blocked!' if unsafe
    """
    # Define allowed built-in functions and operations
    allowed_builtins = {
        'abs': abs,
        'all': all,
        'any': any,
        'bool': bool,
        'chr': chr,
        'divmod': divmod,
        'enumerate': enumerate,
        'float': float,
        'int': int,
        'len': len,
        'max': max,
        'min': min,
        'ord': ord,
        'pow': pow,
        'range': range,
        'round': round,
        'str': str,
        'sum': sum,
        'zip': zip,
        '__builtins__': {},
        'True': True,
        'False': False,
        'None': None
    }
    
    # List of potentially dangerous keywords and functions
    dangerous_keywords = [
        '__import__', 'import', 'from', 'exec', 'eval', 'compile',
        'open', 'file', 'input', 'raw_input', 'globals', 'locals',
        'vars', 'dir', 'delattr', 'setattr', 'getattr', 'hasattr',
        'reload', 'type', 'id', 'help', 'print', 'breakpoint',
        'classmethod', 'staticmethod', 'property', 'super',
        'lambda', 'yield', 'class', 'def', 'del', 'global',
        'nonlocal', 'assert', 'with', 'as', 'raise', 'try',
        'except', 'finally', 'async', 'await', '__'
    ]
    
    try:
        # Check for dangerous keywords
        code_lower = code_script.lower()
        for keyword in dangerous_keywords:
            if keyword in code_lower:
                return "Execution Blocked!"
        
        # Create a restricted execution environment
        restricted_globals = allowed_builtins.copy()
        restricted_locals = {}
        
        # Execute the code in the restricted environment
        exec(code_script, restricted_globals, restricted_locals)
        
        # If the script doesn't explicitly return a value, 
        # return the last expression result if available
        if '_' in restricted_locals:
            return restricted_locals['_']
        elif restricted_locals:
            # Return the last assigned variable
            return list(restricted_locals.values())[-1]
        else:
            # Try to evaluate as an expression
            return eval(code_script, restricted_globals)
            
    except Exception:
        return "Execution Blocked!"


# Test the function
if __name__ == "__main__":
    # Test cases
    test_scripts = [
        "2 + 3",
        "result = 10 * 5",
        "x = 5\ny = 10\nx + y",
        "'hello' + ' ' + 'world'",
        "text = 'Python'\ntext.upper()",
        "numbers = [1, 2, 3, 4, 5]\nsum(numbers)",
        "import os",
        "__import__('os')",
        "open('/etc/passwd', 'r')",
        "exec('print(1)')",
        "eval('2+2')",
        "globals()",
        "[i**2 for i in range(5)]",
        "max([1, 5, 3, 9, 2])",
        "'test' * 3",
        "str(42) + ' is the answer'",
        "len('hello world')",
        "round(3.14159, 2)",
        "abs(-10)",
        "divmod(17, 5)"
    ]
    
    for script in test_scripts:
        result = process_user_code(script)
        print(f"Script: {repr(script)}")
        print(f"Result: {result}")
        print("-" * 50)
