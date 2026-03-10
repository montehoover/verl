import ast
import re

def filter_unsafe_operations(script):
    """
    Check if a Python script contains only safe operations (basic arithmetic and string manipulations).
    Returns True if safe, False if potentially unsafe operations are detected.
    """
    # First check for obvious dangerous patterns using regex
    dangerous_patterns = [
        r'\b(import|from)\s+\w+',  # Import statements
        r'\b(__import__|eval|exec|compile|open|file|input|raw_input)\s*\(',  # Dangerous built-ins
        r'\b(os|sys|subprocess|socket|requests|urllib)\b',  # Common dangerous modules
        r'\b(globals|locals|vars|dir)\s*\(',  # Introspection functions
        r'__\w+__',  # Dunder methods/attributes
        r'\bgetattr\s*\(',  # Dynamic attribute access
        r'\bsetattr\s*\(',  # Dynamic attribute setting
        r'\bdelattr\s*\(',  # Dynamic attribute deletion
        r'\b(lambda)\s+',  # Lambda functions
        r'\bdef\s+\w+\s*\(',  # Function definitions
        r'\bclass\s+\w+',  # Class definitions
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, script, re.IGNORECASE):
            return False
    
    # Parse the AST to check for allowed operations
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return False
    
    # Define allowed node types for safe operations
    allowed_nodes = {
        # Literals
        ast.Constant, ast.Num, ast.Str, ast.Bytes,
        ast.NameConstant, ast.List, ast.Tuple,
        
        # Variables
        ast.Name, ast.Load, ast.Store,
        
        # Operators
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Mod, ast.Pow, ast.USub, ast.UAdd,
        
        # Comparisons
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Is, ast.IsNot, ast.In, ast.NotIn,
        
        # Boolean operators
        ast.And, ast.Or, ast.Not,
        
        # Expressions
        ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
        ast.IfExp,
        
        # Statements
        ast.Assign, ast.AugAssign, ast.Expr,
        ast.If, ast.For, ast.While, ast.Break, ast.Continue,
        ast.Pass,
        
        # Containers
        ast.ListComp, ast.Subscript, ast.Index, ast.Slice,
        
        # Module level
        ast.Module,
    }
    
    # Walk through all nodes in the AST
    for node in ast.walk(tree):
        if type(node) not in allowed_nodes:
            # Check for specific allowed calls
            if isinstance(node, ast.Call):
                # Allow only specific string methods
                if isinstance(node.func, ast.Attribute):
                    allowed_methods = {
                        'upper', 'lower', 'strip', 'lstrip', 'rstrip',
                        'replace', 'split', 'join', 'startswith', 'endswith',
                        'find', 'rfind', 'index', 'rindex', 'count',
                        'capitalize', 'title', 'swapcase', 'center',
                        'ljust', 'rjust', 'zfill', 'format',
                        'isalpha', 'isdigit', 'isalnum', 'isspace',
                        'isupper', 'islower', 'istitle',
                    }
                    if node.func.attr not in allowed_methods:
                        return False
                # Allow built-in functions for type conversion
                elif isinstance(node.func, ast.Name):
                    allowed_builtins = {'int', 'float', 'str', 'bool', 'len', 'range', 'min', 'max', 'sum'}
                    if node.func.id not in allowed_builtins:
                        return False
                else:
                    return False
            else:
                return False
    
    # Additional check: ensure no attribute access except for allowed string methods
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            # Only allow if it's part of a method call we already validated
            parent = None
            for parent_node in ast.walk(tree):
                for child in ast.iter_child_nodes(parent_node):
                    if child == node and isinstance(parent_node, ast.Call):
                        parent = parent_node
                        break
            if parent is None:
                return False
    
    return True


def evaluate_safe_script(script):
    """
    Execute a script that has been confirmed to be safe.
    Returns the result of the script's execution or an error message.
    """
    # Create a restricted namespace with only allowed builtins
    safe_builtins = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'len': len,
        'range': range,
        'min': min,
        'max': max,
        'sum': sum,
        'True': True,
        'False': False,
        'None': None,
    }
    
    # Create a namespace for execution
    namespace = {'__builtins__': safe_builtins}
    
    try:
        # Compile and execute the script
        compiled = compile(script, '<string>', 'exec')
        exec(compiled, namespace)
        
        # Find the last expression to return its value
        tree = ast.parse(script)
        last_expr = None
        
        for node in tree.body:
            if isinstance(node, ast.Expr):
                last_expr = node
        
        # If there's a last expression, evaluate and return it
        if last_expr:
            expr_code = compile(ast.Expression(last_expr.value), '<string>', 'eval')
            result = eval(expr_code, namespace)
            return result
        else:
            # If no expression to evaluate, check for any variables created
            # Remove built-ins from namespace to see what was created
            created_vars = {k: v for k, v in namespace.items() if k not in safe_builtins and not k.startswith('__')}
            if created_vars:
                # If only one variable, return its value
                if len(created_vars) == 1:
                    return list(created_vars.values())[0]
                # If multiple variables, return the namespace
                return created_vars
            else:
                return None
                
    except Exception as e:
        return f"Error: Problem with script execution - {type(e).__name__}: {str(e)}"


def execute_user_code(script_code):
    """
    Process a user-supplied script, checking for safety before execution.
    Returns the result of the script if safe, or 'Execution Blocked!' if unsafe.
    """
    # Check if the script contains only safe operations
    if not filter_unsafe_operations(script_code):
        return 'Execution Blocked!'
    
    # If safe, evaluate the script
    return evaluate_safe_script(script_code)
