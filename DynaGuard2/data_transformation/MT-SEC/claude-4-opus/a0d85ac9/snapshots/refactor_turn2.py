import ast
import re


def get_allowed_names():
    """
    Return a dictionary of allowed names/functions for safe evaluation.
    These are basic mathematical and logical operations that pose no security risk.
    """
    return {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'True': True,
        'False': False,
        'None': None
    }


def get_dangerous_keywords():
    """
    Return a list of keywords that could pose security risks if allowed in expressions.
    These include import statements, file operations, and introspection functions.
    """
    return [
        '__', 'import', 'exec', 'eval', 'open', 'file', 'input',
        'compile', 'globals', 'locals', 'vars', 'dir', 'getattr',
        'setattr', 'delattr', 'type', 'help', 'breakpoint', 'exit',
        'quit', 'credits', 'license', 'copyright', 'print'
    ]


def contains_dangerous_keywords(expression):
    """
    Check if the expression contains any dangerous keywords.
    
    Args:
        expression: str - The expression to check
        
    Returns:
        bool - True if dangerous keywords found, False otherwise
    """
    expr_lower = expression.lower()
    dangerous_keywords = get_dangerous_keywords()
    
    for keyword in dangerous_keywords:
        if keyword in expr_lower:
            return True
    return False


def validate_function_calls(expression):
    """
    Validate that only allowed functions are called in the expression.
    
    Args:
        expression: str - The expression to check
        
    Returns:
        bool - True if all function calls are valid, False otherwise
    """
    if '(' not in expression:
        return True
    
    # Extract function names from the expression
    func_pattern = r'(\w+)\s*\('
    matches = re.findall(func_pattern, expression)
    allowed_names = get_allowed_names()
    
    for match in matches:
        if match not in allowed_names:
            return False
    return True


def get_allowed_ast_nodes():
    """
    Return a tuple of allowed AST node types for safe expression evaluation.
    These nodes represent basic arithmetic, logical, and comparison operations.
    """
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare,
        ast.Constant, ast.Name, ast.Load, ast.Store,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Mod, ast.Pow, ast.LShift, ast.RShift,
        ast.BitOr, ast.BitXor, ast.BitAnd, ast.And, ast.Or,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Is, ast.IsNot, ast.In, ast.NotIn, ast.Not,
        ast.Invert, ast.UAdd, ast.USub, ast.Call, ast.keyword,
        ast.IfExp, ast.BoolOp
    )
    
    # Add compatibility for Python 3.7 and earlier
    if hasattr(ast, 'Num'):
        allowed_nodes = allowed_nodes + (ast.Num, ast.Str, ast.NameConstant)
    
    return allowed_nodes


def validate_ast_nodes(tree):
    """
    Validate that the AST contains only allowed node types.
    
    Args:
        tree: ast.AST - The parsed AST tree to validate
        
    Returns:
        bool - True if all nodes are valid, False otherwise
    """
    allowed_nodes = get_allowed_ast_nodes()
    allowed_names = get_allowed_names()
    
    # Walk through all nodes in the AST
    for node in ast.walk(tree):
        # Check if node type is allowed
        if not isinstance(node, allowed_nodes):
            return False
        
        # Validate Name nodes
        if isinstance(node, ast.Name):
            if node.id not in allowed_names and not node.id.replace('_', '').isalnum():
                return False
        
        # Validate Call nodes - ensure they only call allowed functions
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in allowed_names:
                    return False
            else:
                return False
    
    return True


def parse_and_validate_expression(expression):
    """
    Parse the expression and validate its safety.
    
    Args:
        expression: str - The expression to parse and validate
        
    Returns:
        tuple - (success: bool, compiled_code: code object or None, error_msg: str or None)
    """
    # Check for dangerous keywords
    if contains_dangerous_keywords(expression):
        return False, None, "Expression contains dangerous keywords"
    
    # Validate function calls
    if not validate_function_calls(expression):
        return False, None, "Expression contains disallowed function calls"
    
    # Try to compile the expression
    try:
        compiled = compile(expression, '<user_input>', 'eval')
    except SyntaxError:
        return False, None, "Invalid syntax"
    
    # Parse and validate AST
    try:
        tree = ast.parse(expression, mode='eval')
    except:
        return False, None, "Failed to parse expression"
    
    # Validate AST nodes
    if not validate_ast_nodes(tree):
        return False, None, "Expression contains disallowed operations"
    
    return True, compiled, None


def evaluate_safe_expression(compiled_code):
    """
    Safely evaluate a compiled expression with a restricted namespace.
    
    Args:
        compiled_code: code - The compiled code object to evaluate
        
    Returns:
        tuple - (success: bool, result: any or None)
    """
    allowed_names = get_allowed_names()
    
    try:
        # Evaluate with empty builtins and only allowed names
        result = eval(compiled_code, {"__builtins__": {}}, allowed_names)
        return True, result
    except Exception:
        return False, None


def calculate_user_input(input_expr: str) -> str:
    """
    Safely evaluate arithmetic and logical expressions from user input.
    
    This function validates the input expression for security risks before evaluation.
    It only allows basic arithmetic, logical operations, and a few safe built-in functions.
    
    Args:
        input_expr: str - A string representing the user-supplied expression
        
    Returns:
        str - The outcome of the evaluated expression or 'Processing Failed!'
    """
    try:
        # Step 1: Parse and validate the expression
        is_valid, compiled_code, error_msg = parse_and_validate_expression(input_expr)
        
        if not is_valid:
            return 'Processing Failed!'
        
        # Step 2: Evaluate the validated expression
        success, result = evaluate_safe_expression(compiled_code)
        
        if not success:
            return 'Processing Failed!'
        
        # Step 3: Convert result to string and return
        return str(result)
        
    except Exception:
        # Catch any unexpected errors
        return 'Processing Failed!'
