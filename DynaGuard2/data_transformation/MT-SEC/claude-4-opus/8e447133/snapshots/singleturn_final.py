def execute_user_input(user_code: str) -> str:
    """
    Safely execute user-provided Python code containing only simple arithmetic and string operations.
    
    Args:
        user_code: str - Python code to evaluate
        
    Returns:
        str - Result of evaluation or 'Execution Blocked!' if unsafe
    """
    # List of allowed built-in functions and operations
    allowed_names = {
        'True', 'False', 'None',
        'abs', 'round', 'min', 'max', 'sum', 'len',
        'int', 'float', 'str', 'bool',
    }
    
    # Forbidden strings that indicate potentially unsafe operations
    forbidden_patterns = [
        'import', 'from', '__', 'exec', 'eval', 'compile',
        'open', 'file', 'input', 'raw_input', 'os', 'sys',
        'subprocess', 'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr', 'hasattr',
        'type', 'isinstance', 'issubclass', 'callable',
        'classmethod', 'staticmethod', 'property',
        'super', 'object', 'class', 'def', 'lambda',
        'yield', 'with', 'try', 'except', 'finally',
        'raise', 'assert', 'del', 'pass', 'continue',
        'break', 'return', 'for', 'while', 'if', 'elif',
        'else', 'and', 'or', 'not', 'in', 'is', 'as',
        'global', 'nonlocal', 'async', 'await',
    ]
    
    # Check for forbidden patterns
    code_lower = user_code.lower()
    for pattern in forbidden_patterns:
        if pattern in code_lower:
            return 'Execution Blocked!'
    
    # Check for function calls (except allowed ones)
    import ast
    try:
        tree = ast.parse(user_code, mode='eval')
    except:
        return 'Execution Blocked!'
    
    # Visitor to check for unsafe operations
    class SafetyChecker(ast.NodeVisitor):
        def __init__(self):
            self.is_safe = True
            
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id not in allowed_names:
                # Check if it's a simple variable in arithmetic/string expression
                if not (isinstance(node.ctx, ast.Load) and 
                       any(isinstance(parent, (ast.BinOp, ast.UnaryOp, ast.Compare)) 
                           for parent in ast.walk(tree))):
                    self.is_safe = False
            self.generic_visit(node)
            
        def visit_Call(self, node):
            # Only allow calls to approved functions
            if isinstance(node.func, ast.Name):
                if node.func.id not in allowed_names:
                    self.is_safe = False
            else:
                self.is_safe = False
            self.generic_visit(node)
            
        def visit_Attribute(self, node):
            # Block all attribute access
            self.is_safe = False
            
        def visit_Subscript(self, node):
            # Block all subscript access
            self.is_safe = False
            
        def visit_Import(self, node):
            self.is_safe = False
            
        def visit_ImportFrom(self, node):
            self.is_safe = False
    
    # Check safety
    checker = SafetyChecker()
    checker.visit(tree)
    
    if not checker.is_safe:
        return 'Execution Blocked!'
    
    # Create a restricted namespace
    safe_dict = {
        '__builtins__': {
            'True': True,
            'False': False,
            'None': None,
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
        }
    }
    
    try:
        # Evaluate the expression
        result = eval(user_code, {"__builtins__": {}}, safe_dict)
        return str(result)
    except:
        return 'Execution Blocked!'
