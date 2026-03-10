def safely_execute_expression(expression_code: str) -> str:
    """
    Safely executes a Python mathematical expression and returns the result.
    
    Args:
        expression_code: A string representing a user-input Python expression for calculation
        
    Returns:
        Either the result of the secure calculation or an error message flagging unsafe code
    """
    # Define allowed names for mathematical operations
    allowed_names = {
        # Math functions
        'abs', 'round', 'min', 'max', 'sum', 'pow', 'divmod',
        # Constants
        'True', 'False', 'None',
        # Type conversions
        'int', 'float', 'complex', 'bool',
        # Math module functions (if needed)
        'sqrt', 'sin', 'cos', 'tan', 'log', 'exp', 'pi', 'e'
    }
    
    # Define allowed node types for the AST
    import ast
    allowed_node_types = (
        ast.Module, ast.Expr, ast.Load, ast.Store,
        ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.FloorDiv, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor,
        ast.BitAnd, ast.MatMult, ast.UAdd, ast.USub, ast.Not, ast.Invert,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Is, ast.IsNot, ast.In, ast.NotIn, ast.And, ast.Or,
        ast.Constant, ast.Num, ast.Str, ast.Name,
        ast.List, ast.Tuple, ast.Set, ast.Dict,
        ast.IfExp,  # Allow ternary expressions
    )
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression_code, mode='eval')
        
        # Check for unsafe nodes
        for node in ast.walk(tree):
            # Check if node type is allowed
            if not isinstance(node, allowed_node_types):
                return "Unsafe Code Detected!"
            
            # Check Name nodes for allowed names
            if isinstance(node, ast.Name) and node.id not in allowed_names:
                # Allow single letter variables for simple math
                if not (len(node.id) == 1 and node.id.isalpha()):
                    return "Unsafe Code Detected!"
            
            # Prevent attribute access (e.g., obj.method())
            if isinstance(node, ast.Attribute):
                return "Unsafe Code Detected!"
            
            # Prevent function calls except for allowed built-ins
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in allowed_names:
                    continue
                else:
                    return "Unsafe Code Detected!"
            
            # Prevent imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return "Unsafe Code Detected!"
            
            # Prevent function/class definitions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, 
                               ast.ClassDef, ast.Lambda)):
                return "Unsafe Code Detected!"
            
            # Prevent dangerous operations
            if isinstance(node, (ast.Exec, ast.Global, ast.Nonlocal)):
                return "Unsafe Code Detected!"
        
        # Create a safe namespace with only allowed builtins
        safe_namespace = {
            '__builtins__': {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow, 'divmod': divmod,
                'int': int, 'float': float, 'complex': complex, 'bool': bool,
                'True': True, 'False': False, 'None': None,
            }
        }
        
        # Import math functions if needed
        import math
        math_functions = {
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
            'tan': math.tan, 'log': math.log, 'exp': math.exp,
            'pi': math.pi, 'e': math.e
        }
        safe_namespace.update(math_functions)
        
        # Compile and evaluate the expression
        compiled_code = compile(tree, '<string>', 'eval')
        result = eval(compiled_code, {"__builtins__": {}}, safe_namespace)
        
        # Return the result as a string
        return str(result)
        
    except Exception:
        # Any error during parsing or evaluation is treated as unsafe
        return "Unsafe Code Detected!"
