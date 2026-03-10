import re

def get_allowed_names():
    """Returns a dictionary of allowed built-in functions for safe execution."""
    return {
        # Basic arithmetic operations
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'len': len,
        # Type conversions
        'int': int,
        'float': float,
        'str': str,
    }

def get_dangerous_keywords():
    """Returns a list of keywords that should block execution."""
    return [
        'import', 'from', '__import__', 'exec', 'eval', 'compile',
        'open', 'file', 'input', 'raw_input', 'globals', 'locals',
        'vars', 'dir', 'delattr', 'setattr', 'getattr', 'hasattr',
        '__', 'lambda', 'def', 'class', 'for', 'while', 'with',
        'try', 'except', 'finally', 'raise', 'assert', 'del',
        'global', 'nonlocal', 'yield', 'async', 'await',
        'breakpoint', 'help', 'quit', 'exit', 'copyright', 'credits',
        'license'
    ]

def get_allowed_string_methods():
    """Returns a set of allowed string methods."""
    return {
        'upper', 'lower', 'strip', 'lstrip', 'rstrip',
        'replace', 'split', 'join', 'startswith', 'endswith',
        'find', 'count', 'isdigit', 'isalpha', 'isalnum',
        'title', 'capitalize', 'swapcase'
    }

def contains_dangerous_keywords(user_code):
    """Checks if the code contains any dangerous keywords."""
    code_lower = user_code.lower()
    dangerous_keywords = get_dangerous_keywords()
    
    for keyword in dangerous_keywords:
        if keyword in code_lower:
            return True
    return False

def contains_unsafe_methods(user_code):
    """Checks if the code contains unsafe method calls."""
    if '.' not in user_code:
        return False
    
    allowed_methods = get_allowed_string_methods()
    method_calls = re.findall(r'\.(\w+)\s*\(', user_code)
    
    for method in method_calls:
        if method not in allowed_methods:
            return True
    return False

def validate_user_code(user_code):
    """Validates user code for safety. Returns True if safe, False otherwise."""
    if contains_dangerous_keywords(user_code):
        return False
    
    if contains_unsafe_methods(user_code):
        return False
    
    return True

def execute_as_expression(user_code, allowed_names):
    """Attempts to execute code as an expression. Returns (success, result)."""
    try:
        compiled_code = compile(user_code, '<user_input>', 'eval')
        restricted_globals = {'__builtins__': allowed_names}
        result = eval(compiled_code, restricted_globals, {})
        return True, str(result)
    except:
        return False, None

def execute_as_statement(user_code, allowed_names):
    """Attempts to execute code as a statement. Returns (success, result)."""
    try:
        compiled_code = compile(user_code, '<user_input>', 'exec')
        restricted_globals = {'__builtins__': allowed_names}
        local_namespace = {}
        
        exec(compiled_code, restricted_globals, local_namespace)
        
        if local_namespace:
            last_value = list(local_namespace.values())[-1]
            return True, str(last_value)
        else:
            return True, ''
    except:
        return False, None

def execute_user_input(user_code: str) -> str:
    """
    Processes a provided script string, evaluates simple arithmetic and string operations,
    and safely returns the corresponding result.
    
    Args:
        user_code: The content of a Python script provided by a user
        
    Returns:
        The result after script evaluation or 'Execution Blocked!' if unsafe
    """
    # Validate the code first
    if not validate_user_code(user_code):
        return 'Execution Blocked!'
    
    allowed_names = get_allowed_names()
    
    # Try to execute as expression first
    success, result = execute_as_expression(user_code, allowed_names)
    if success:
        return result
    
    # If expression fails, try as statement
    success, result = execute_as_statement(user_code, allowed_names)
    if success:
        return result
    
    # If both fail, block execution
    return 'Execution Blocked!'
