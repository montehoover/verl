import ast

def execute_python_code(snippet_code):
    """
    Safely execute a string of Python code.
    
    Args:
        snippet_code: A string containing Python code to execute
    
    Returns:
        The resulting value if the code produces output, or None otherwise
        
    Raises:
        ValueError: If the code attempts forbidden operations or is invalid
    """
    # Parse the code to check for syntax errors
    try:
        tree = ast.parse(snippet_code)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}")
    
    # Check for forbidden operations
    forbidden_names = {'__import__', 'open', 'file', 'input', 'raw_input', 
                      'compile', 'execfile', 'eval', 'exec'}
    
    class ForbiddenChecker(ast.NodeVisitor):
        def visit_Name(self, node):
            if node.id in forbidden_names:
                raise ValueError(f"Forbidden operation: {node.id}")
            self.generic_visit(node)
            
        def visit_Import(self, node):
            raise ValueError("Import statements are not allowed")
            
        def visit_ImportFrom(self, node):
            raise ValueError("Import statements are not allowed")
            
        def visit_FunctionDef(self, node):
            if node.name in forbidden_names:
                raise ValueError(f"Forbidden function name: {node.name}")
            self.generic_visit(node)
    
    checker = ForbiddenChecker()
    checker.visit(tree)
    
    # Create a restricted environment
    safe_globals = {
        '__builtins__': {
            'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
            'chr': chr, 'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
            'filter': filter, 'float': float, 'format': format, 'hex': hex,
            'int': int, 'isinstance': isinstance, 'len': len, 'list': list,
            'map': map, 'max': max, 'min': min, 'oct': oct, 'ord': ord,
            'pow': pow, 'print': print, 'range': range, 'reversed': reversed,
            'round': round, 'set': set, 'sorted': sorted, 'str': str,
            'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip
        }
    }
    
    local_vars = {}
    
    # Execute the code
    try:
        exec(compile(tree, '<string>', 'exec'), safe_globals, local_vars)
    except Exception as e:
        raise ValueError(f"Error executing code: {e}")
    
    # Try to get the result of the last expression
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        try:
            last_expr = ast.Expression(tree.body[-1].value)
            result = eval(compile(last_expr, '<string>', 'eval'), safe_globals, local_vars)
            return result
        except:
            pass
    
    return None
