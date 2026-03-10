import ast

def process_code_input(code_input: str):
    """
    Execute a Python code snippet safely with strict security rules.
    
    Args:
        code_input: str - A string containing the Python code to execute
        
    Returns:
        None if there is no output from the code, or the result if any
        
    Raises:
        ValueError: If the code tries to execute any disallowed operations
    """
    
    # Define allowed AST node types for safe execution
    ALLOWED_NODE_TYPES = {
        # Literals and basic types
        ast.Constant, ast.Num, ast.Str, ast.Bytes, ast.NameConstant,
        
        # Variables and names
        ast.Name, ast.Load, ast.Store,
        
        # Basic operations
        ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
        ast.And, ast.Or, ast.Not,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Is, ast.IsNot, ast.In, ast.NotIn,
        ast.UAdd, ast.USub, ast.Invert,
        
        # Collections
        ast.List, ast.Tuple, ast.Set, ast.Dict,
        
        # Control flow (limited)
        ast.If, ast.For, ast.While, ast.Break, ast.Continue,
        ast.Assign, ast.AugAssign, ast.AnnAssign,
        
        # Expressions
        ast.Expr, ast.Expression,
        
        # Comprehensions
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
        ast.comprehension,
        
        # Indexing and slicing
        ast.Subscript, ast.Index, ast.Slice,
        
        # Module level
        ast.Module,
        
        # Other safe operations
        ast.Pass, ast.Ellipsis,
    }
    
    # Disallowed built-in functions
    DISALLOWED_BUILTINS = {
        'eval', 'exec', 'compile', '__import__', 'open', 'input',
        'file', 'execfile', 'reload', 'vars', 'globals', 'locals',
        'dir', 'help', 'quit', 'exit', 'copyright', 'credits', 'license'
    }
    
    try:
        # Parse the code into an AST
        tree = ast.parse(code_input, mode='exec')
        
        # Check for disallowed operations
        for node in ast.walk(tree):
            # Check if node type is allowed
            if type(node) not in ALLOWED_NODE_TYPES:
                # Special handling for some nodes
                if isinstance(node, ast.Call):
                    # Check if it's calling a disallowed builtin
                    if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_BUILTINS:
                        raise ValueError(f"Disallowed function call: {node.func.id}")
                    # Allow only safe built-in functions
                    if isinstance(node.func, ast.Name) and node.func.id in ['print', 'len', 'range', 'int', 'float', 'str', 'bool', 'list', 'tuple', 'dict', 'set', 'min', 'max', 'sum', 'abs', 'round', 'sorted', 'reversed', 'enumerate', 'zip', 'map', 'filter', 'any', 'all']:
                        continue
                    raise ValueError(f"Function calls are restricted")
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    raise ValueError("Import statements are not allowed")
                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    raise ValueError("Function definitions are not allowed")
                elif isinstance(node, ast.ClassDef):
                    raise ValueError("Class definitions are not allowed")
                elif isinstance(node, ast.Delete):
                    raise ValueError("Delete statements are not allowed")
                elif isinstance(node, ast.With) or isinstance(node, ast.AsyncWith):
                    raise ValueError("Context managers (with statements) are not allowed")
                elif isinstance(node, ast.Raise):
                    raise ValueError("Raise statements are not allowed")
                elif isinstance(node, ast.Try):
                    raise ValueError("Try/except blocks are not allowed")
                elif isinstance(node, ast.Assert):
                    raise ValueError("Assert statements are not allowed")
                elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
                    raise ValueError("Global/nonlocal declarations are not allowed")
                elif isinstance(node, ast.Lambda):
                    raise ValueError("Lambda functions are not allowed")
                elif isinstance(node, ast.Yield) or isinstance(node, ast.YieldFrom):
                    raise ValueError("Yield statements are not allowed")
                elif isinstance(node, ast.Await):
                    raise ValueError("Await expressions are not allowed")
                else:
                    raise ValueError(f"Disallowed operation: {type(node).__name__}")
            
            # Check for attribute access (dangerous)
            if isinstance(node, ast.Attribute):
                raise ValueError("Attribute access is not allowed")
        
        # Create a restricted namespace
        safe_namespace = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'int': int,
                'float': float,
                'str': str,
                'bool': bool,
                'list': list,
                'tuple': tuple,
                'dict': dict,
                'set': set,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'reversed': reversed,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'any': any,
                'all': all,
                'True': True,
                'False': False,
                'None': None,
            }
        }
        
        # Compile and execute the code
        compiled_code = compile(tree, '<string>', 'exec')
        
        # Capture any expression result
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            # If the last statement is an expression, evaluate it separately
            expr_tree = ast.Expression(body=tree.body[-1].value)
            expr_code = compile(expr_tree, '<string>', 'eval')
            
            # Execute all but the last statement
            if len(tree.body) > 1:
                exec_tree = ast.Module(body=tree.body[:-1], type_ignores=[])
                exec_code = compile(exec_tree, '<string>', 'exec')
                exec(exec_code, safe_namespace)
            
            # Evaluate the last expression
            last_expr = eval(expr_code, safe_namespace)
        else:
            # Execute normally
            exec(compiled_code, safe_namespace)
        
        # Return the result of the last expression if any
        return last_expr
        
    except SyntaxError as e:
        raise ValueError(f"Syntax error in code: {e}")
    except Exception as e:
        # Re-raise as ValueError to maintain consistent error type
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error executing code: {e}")
