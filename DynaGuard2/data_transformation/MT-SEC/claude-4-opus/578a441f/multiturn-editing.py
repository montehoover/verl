import re
import ast

def analyze_script(script: str) -> list:
    # Define security-sensitive characters
    security_sensitive = ['<', '>', '&', '|', ';', '$', '`', '\\', '"', "'", '\n', '\r', '\0']
    
    # Check for security-sensitive characters
    for char in security_sensitive:
        if char in script:
            raise ValueError(f"Security-sensitive character '{char}' found in script")
    
    # Define token pattern - matches words, numbers, and common operators
    token_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|\+|\-|\*|\/|\=|\(|\)|\{|\}|\[|\]|\,|\.'
    
    # Find all tokens
    tokens = re.findall(token_pattern, script)
    
    # Define approved syntax elements
    approved_keywords = {'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'return', 'import', 'from', 'as', 'in', 'is', 'and', 'or', 'not', 'True', 'False', 'None', 'pass', 'break', 'continue', 'try', 'except', 'finally', 'with', 'lambda', 'yield'}
    approved_operators = {'+', '-', '*', '/', '=', '(', ')', '{', '}', '[', ']', ',', '.'}
    
    # Validate and filter tokens
    validated_tokens = []
    for token in tokens:
        # Check if token is a number
        if token.isdigit():
            validated_tokens.append(token)
        # Check if token is an approved operator
        elif token in approved_operators:
            validated_tokens.append(token)
        # Check if token is an approved keyword
        elif token in approved_keywords:
            validated_tokens.append(token)
        # Check if token is a valid identifier (variable/function name)
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token) and len(token) <= 100:
            validated_tokens.append(token)
        else:
            raise ValueError(f"Invalid token: '{token}'")
    
    # Verify all characters in script are part of valid tokens or whitespace
    reconstructed = ''
    for token in tokens:
        reconstructed += token
    
    # Remove all whitespace from original script for comparison
    script_no_whitespace = re.sub(r'\s+', '', script)
    reconstructed_no_whitespace = re.sub(r'\s+', '', reconstructed)
    
    if script_no_whitespace != reconstructed_no_whitespace:
        raise ValueError("Script contains invalid tokens")
    
    return validated_tokens


def run_user_script(user_script: str):
    # Parse the script into an AST
    try:
        tree = ast.parse(user_script, mode='exec')
    except SyntaxError as e:
        raise ValueError(f"Syntax error in script: {e}")
    
    # Define disallowed AST node types
    disallowed_nodes = {
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.With,
        ast.AsyncWith,
        ast.Raise,
        ast.Try,
        ast.Assert,
        ast.Global,
        ast.Nonlocal,
        ast.Yield,
        ast.YieldFrom,
        ast.GeneratorExp,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.comprehension,
        ast.Lambda,
        ast.Call,
    }
    
    # Check for disallowed operations
    for node in ast.walk(tree):
        if type(node) in disallowed_nodes:
            raise ValueError(f"Disallowed operation: {type(node).__name__}")
        
        # Check for attribute access (could be dangerous)
        if isinstance(node, ast.Attribute):
            raise ValueError("Attribute access is not allowed")
        
        # Check for subscript access that could be dangerous
        if isinstance(node, ast.Subscript):
            raise ValueError("Subscript access is not allowed")
    
    # Create a restricted environment
    safe_globals = {
        '__builtins__': {
            'True': True,
            'False': False,
            'None': None,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'tuple': tuple,
            'dict': dict,
            'set': set,
            'range': range,
        }
    }
    
    safe_locals = {}
    
    # Compile and execute the code
    try:
        compiled_code = compile(tree, filename='<user_script>', mode='exec')
        exec(compiled_code, safe_globals, safe_locals)
        
        # Check if there's a result to return
        if safe_locals:
            # Return the last assigned value if any
            last_value = list(safe_locals.values())[-1]
            return last_value
        else:
            return None
            
    except Exception as e:
        raise ValueError(f"Error executing script: {e}")
