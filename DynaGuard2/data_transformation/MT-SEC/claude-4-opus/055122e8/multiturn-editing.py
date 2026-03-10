import ast

def execute_python_script(python_code):
    """
    Execute a user-provided Python script safely.
    
    Args:
        python_code (str): A string containing Python code to execute
    
    Returns:
        The result of the executed script, or None if there's no result
    
    Raises:
        ValueError: If the script contains invalid syntax or prohibited operations
    """
    try:
        # Parse the code to check for syntax errors
        tree = ast.parse(python_code, mode='exec')
        
        # Check for potentially harmful operations
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Import statements are not allowed")
            if isinstance(node, ast.Name) and node.id in ['__import__', 'open', 'exec', 'eval', 'compile']:
                raise ValueError(f"Use of '{node.id}' is not allowed")
            if isinstance(node, ast.Attribute):
                if (isinstance(node.value, ast.Name) and node.value.id == 'os') or \
                   (node.attr in ['__globals__', '__locals__', '__builtins__']):
                    raise ValueError("Access to system internals is not allowed")
        
        # Execute the code in a restricted namespace
        namespace = {}
        exec(compile(tree, '<string>', 'exec'), namespace)
        
        # Return the last expression's value if any
        if python_code.strip().split('\n')[-1].strip() and not python_code.strip().endswith(':'):
            try:
                result_tree = ast.parse(python_code.strip().split('\n')[-1], mode='eval')
                return eval(compile(result_tree, '<string>', 'eval'), namespace)
            except:
                return None
        
        return None
        
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")
    except Exception as e:
        raise ValueError(f"Error executing script: {e}")
