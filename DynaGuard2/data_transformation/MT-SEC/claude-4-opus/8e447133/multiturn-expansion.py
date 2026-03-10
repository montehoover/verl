import ast
import re

def parse_script_operations(script):
    """
    Parse a user-supplied script and return a list of valid operations.
    Only allows safe arithmetic and string operations.
    
    Args:
        script (str): The user-supplied script to parse
        
    Returns:
        list: A list of valid operations found in the script
        
    Raises:
        ValueError: If the script contains invalid or unsafe operations
    """
    operations = []
    
    # Define allowed operation types
    allowed_ast_types = {
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.And, ast.Or, ast.Not,
        ast.UAdd, ast.USub, ast.Invert
    }
    
    allowed_node_types = {
        ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
        ast.Constant, ast.Num, ast.Str, ast.Name, ast.Load,
        ast.Expr, ast.Module
    }
    
    # List of forbidden names/functions
    forbidden_names = {
        'eval', 'exec', 'compile', '__import__', 'open', 'input',
        'file', 'execfile', 'globals', 'locals', 'vars',
        'getattr', 'setattr', 'delattr', 'hasattr',
        'dir', 'help', 'reload', 'raw_input', 'apply', 'buffer',
        'coerce', 'intern'
    }
    
    try:
        # Parse the script into an AST
        tree = ast.parse(script, mode='exec')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")
    
    def validate_node(node):
        """Recursively validate AST nodes"""
        node_type = type(node)
        
        # Check if node type is allowed
        if node_type not in allowed_node_types:
            if hasattr(ast, node_type.__name__):
                raise ValueError(f"Operation type '{node_type.__name__}' is not allowed")
        
        # Check for forbidden names
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            raise ValueError(f"Forbidden name '{node.id}' detected")
        
        # Check for attribute access (could be dangerous)
        if isinstance(node, ast.Attribute):
            raise ValueError("Attribute access is not allowed")
        
        # Check for function calls (potentially dangerous)
        if isinstance(node, ast.Call):
            raise ValueError("Function calls are not allowed")
        
        # Check for imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements are not allowed")
        
        # Recursively validate child nodes
        for child in ast.iter_child_nodes(node):
            validate_node(child)
    
    def extract_operation(node):
        """Extract operation description from AST node"""
        if isinstance(node, ast.BinOp):
            op_type = type(node.op).__name__
            return f"Binary operation: {op_type}"
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op).__name__
            return f"Unary operation: {op_type}"
        elif isinstance(node, ast.Compare):
            ops = [type(op).__name__ for op in node.ops]
            return f"Comparison: {', '.join(ops)}"
        elif isinstance(node, ast.BoolOp):
            op_type = type(node.op).__name__
            return f"Boolean operation: {op_type}"
        elif isinstance(node, ast.Expr):
            return extract_operation(node.value)
        return None
    
    # Validate the entire AST
    validate_node(tree)
    
    # Extract operations from the AST
    for node in ast.walk(tree):
        operation = extract_operation(node)
        if operation and operation not in operations:
            operations.append(operation)
    
    return operations


def evaluate_operations(script):
    """
    Safely evaluate a script containing validated operations.
    
    Args:
        script (str): The script to evaluate
        
    Returns:
        The result of the operations, or an error message if unsafe operations are detected
    """
    # Define allowed built-ins for evaluation
    safe_builtins = {
        'abs': abs,
        'bool': bool,
        'float': float,
        'int': int,
        'len': len,
        'max': max,
        'min': min,
        'pow': pow,
        'round': round,
        'str': str,
        'sum': sum,
        'True': True,
        'False': False,
        'None': None
    }
    
    # First validate the script
    try:
        operations = parse_script_operations(script)
    except ValueError as e:
        return f"Safety issue: {e}"
    
    # Create a restricted environment
    restricted_globals = {"__builtins__": safe_builtins}
    restricted_locals = {}
    
    try:
        # Compile the script
        compiled_code = compile(script, '<string>', 'eval')
        
        # Execute in restricted environment
        result = eval(compiled_code, restricted_globals, restricted_locals)
        
        return result
    except Exception as e:
        return f"Safety issue: Evaluation error - {type(e).__name__}: {e}"


def execute_user_input(user_code):
    """
    Process and evaluate user-supplied scripts securely.
    
    Args:
        user_code (str): The user-supplied code to execute
        
    Returns:
        The result of the execution, or 'Execution Blocked!' if unsafe operations are detected
    """
    # Check for empty input
    if not user_code or not user_code.strip():
        return 'Execution Blocked!'
    
    try:
        # Use the evaluate_operations function which includes validation
        result = evaluate_operations(user_code)
        
        # Check if the result indicates a safety issue
        if isinstance(result, str) and result.startswith("Safety issue:"):
            return 'Execution Blocked!'
        
        return result
    except Exception:
        return 'Execution Blocked!'
