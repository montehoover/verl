import ast

def run_code_snippet(snippet: str):
    # Define disallowed operations
    disallowed_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Raise,
        ast.Global,
        ast.Nonlocal,
        ast.Yield,
        ast.YieldFrom,
        ast.GeneratorExp,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.comprehension,
    )
    
    disallowed_names = {
        '__import__', 'eval', 'exec', 'compile', 'open', 'input',
        'breakpoint', 'help', 'dir', 'locals', 'globals', 'vars',
        'setattr', 'delattr', 'getattr', 'hasattr', 'type', 'isinstance',
        'issubclass', 'super', 'property', 'classmethod', 'staticmethod',
        'callable', 'format', 'repr', 'ascii', 'ord', 'chr', 'bin',
        'hex', 'oct', 'abs', 'round', 'divmod', 'pow', 'sorted',
        'enumerate', 'filter', 'map', 'zip', 'reversed', 'iter',
        'next', 'range', 'slice', 'bytes', 'bytearray', 'memoryview',
        'complex', 'bool', 'list', 'tuple', 'set', 'frozenset', 'dict',
        'str', 'int', 'float', 'object', 'all', 'any', 'sum', 'min',
        'max', 'len', 'print', 'id', 'hash'
    }
    
    try:
        # Parse the code to check for disallowed operations
        tree = ast.parse(snippet, mode='exec')
        
        # Check for disallowed nodes
        for node in ast.walk(tree):
            if isinstance(node, disallowed_nodes):
                raise ValueError("Disallowed operation detected")
            
            # Check for disallowed function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in disallowed_names:
                    raise ValueError("Disallowed operation detected")
                elif isinstance(node.func, ast.Attribute):
                    # Check for method calls that could be dangerous
                    if node.func.attr in {'__subclasses__', '__bases__', '__code__', 
                                         '__globals__', '__builtins__', '__class__'}:
                        raise ValueError("Disallowed operation detected")
            
            # Check for attribute access that could be dangerous
            if isinstance(node, ast.Attribute):
                if node.attr in {'__subclasses__', '__bases__', '__code__', 
                               '__globals__', '__builtins__', '__class__', '__dict__',
                               '__module__', '__name__', '__qualname__', '__annotations__'}:
                    raise ValueError("Disallowed operation detected")
            
            # Check for subscript access to builtins
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id in {'__builtins__', 'globals', 'locals'}:
                    raise ValueError("Disallowed operation detected")
        
        # Execute the code in a restricted namespace
        namespace = {
            '__builtins__': {
                'True': True,
                'False': False,
                'None': None,
            }
        }
        
        # Compile and execute
        compiled = compile(tree, '<string>', 'exec')
        exec(compiled, namespace)
        
        # Check if there's a result to return
        # Try to evaluate as an expression to get the last value
        try:
            expr_tree = ast.parse(snippet, mode='eval')
            # Check the expression tree for disallowed operations too
            for node in ast.walk(expr_tree):
                if isinstance(node, disallowed_nodes):
                    raise ValueError("Disallowed operation detected")
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in disallowed_names:
                        raise ValueError("Disallowed operation detected")
                if isinstance(node, ast.Attribute):
                    if node.attr in {'__subclasses__', '__bases__', '__code__', 
                                   '__globals__', '__builtins__', '__class__', '__dict__',
                                   '__module__', '__name__', '__qualname__', '__annotations__'}:
                        raise ValueError("Disallowed operation detected")
            
            result = eval(compile(expr_tree, '<string>', 'eval'), namespace)
            return result
        except:
            # If it's not a single expression, return None
            return None
            
    except SyntaxError:
        raise ValueError("Disallowed operation detected")
    except Exception:
        raise ValueError("Disallowed operation detected")
