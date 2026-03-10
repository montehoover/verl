import ast
import io
import sys

# Define disallowed operations
DISALLOWED_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Delete,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.ExceptHandler,
    ast.Global,
    ast.Nonlocal,
    ast.Yield,
    ast.YieldFrom,
    ast.GeneratorExp,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.comprehension,
    ast.Lambda
)

DISALLOWED_FUNCTIONS = {
    'exec', 'eval', 'compile', 'open', '__import__', 
    'globals', 'locals', 'vars', 'dir', 'getattr', 
    'setattr', 'delattr', 'hasattr', 'type', 'isinstance',
    'issubclass', 'callable', 'classmethod', 'staticmethod',
    'property', 'super'
}

SAFE_BUILTINS = {
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
    'abs': abs,
    'min': min,
    'max': max,
    'sum': sum,
    'round': round,
    'sorted': sorted,
    'reversed': reversed,
    'enumerate': enumerate,
    'zip': zip,
    'map': map,
    'filter': filter,
    'all': all,
    'any': any,
    'True': True,
    'False': False,
    'None': None
}


def parse_code(code_input: str) -> ast.AST:
    """Parse the code input and return the AST."""
    try:
        return ast.parse(code_input, mode='exec')
    except SyntaxError:
        raise ValueError("Invalid Python syntax")


def validate_ast_node(node: ast.AST) -> None:
    """Validate a single AST node for safety."""
    if isinstance(node, DISALLOWED_NODES):
        raise ValueError(f"Disallowed operation: {type(node).__name__}")
    
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in DISALLOWED_FUNCTIONS:
                raise ValueError(f"Disallowed function: {node.func.id}")
    
    if isinstance(node, ast.Attribute):
        if node.attr.startswith('_'):
            raise ValueError("Access to private attributes is not allowed")


def validate_code(tree: ast.AST) -> None:
    """Validate the entire AST tree for safety."""
    for node in ast.walk(tree):
        validate_ast_node(node)


def create_safe_namespace() -> dict:
    """Create a restricted namespace for code execution."""
    return {
        '__builtins__': SAFE_BUILTINS.copy()
    }


def execute_code(tree: ast.AST, safe_namespace: dict) -> tuple[any, str]:
    """Execute the code and return the result and output."""
    old_stdout = sys.stdout
    sys.stdout = output_buffer = io.StringIO()
    
    result = None
    try:
        exec(compile(tree, '<string>', 'exec'), safe_namespace)
    finally:
        sys.stdout = old_stdout
    
    output = output_buffer.getvalue()
    return result, output


def get_last_expression_value(code_input: str, safe_namespace: dict) -> any:
    """Evaluate the last line as an expression if possible."""
    last_line = code_input.strip().split('\n')[-1].strip()
    if not last_line or code_input.strip().endswith(':'):
        return None
    
    try:
        last_expr = ast.parse(last_line, mode='eval')
        # Validate the last expression
        for node in ast.walk(last_expr):
            validate_ast_node(node)
        
        return eval(compile(last_expr, '<string>', 'eval'), safe_namespace)
    except:
        return None


def process_code_input(code_input: str):
    """Process and execute Python code with safety restrictions."""
    # Parse the code
    tree = parse_code(code_input)
    
    # Validate the code
    validate_code(tree)
    
    # Create safe namespace
    safe_namespace = create_safe_namespace()
    
    # Execute the code
    result, output = execute_code(tree, safe_namespace)
    
    # Try to get the value of the last expression
    last_expr_value = get_last_expression_value(code_input, safe_namespace)
    
    # Return result or output
    if last_expr_value is not None:
        return last_expr_value
    elif output:
        return output.rstrip('\n')
    else:
        return None
