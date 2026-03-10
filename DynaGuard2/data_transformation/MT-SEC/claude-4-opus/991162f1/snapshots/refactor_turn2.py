import ast
import operator
import math

# Define allowed operators
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Define allowed functions
ALLOWED_FUNCTIONS = {
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
    'len': len,
    'int': int,
    'float': float,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'sqrt': math.sqrt,
    'log': math.log,
    'log10': math.log10,
    'exp': math.exp,
    'pi': math.pi,
    'e': math.e,
}


def validate_node(node):
    """
    Validate that a node is safe to evaluate.
    
    Args:
        node: An AST node to validate
        
    Returns:
        bool: True if the node is safe, False otherwise
    """
    if isinstance(node, (ast.Num, ast.Constant)):
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float))
        return True
    elif isinstance(node, ast.BinOp):
        return type(node.op) in ALLOWED_OPERATORS
    elif isinstance(node, ast.UnaryOp):
        return type(node.op) in ALLOWED_OPERATORS
    elif isinstance(node, ast.Call):
        return (isinstance(node.func, ast.Name) and 
                node.func.id in ALLOWED_FUNCTIONS)
    elif isinstance(node, ast.Name):
        return (node.id in ALLOWED_FUNCTIONS and 
                isinstance(ALLOWED_FUNCTIONS[node.id], (int, float)))
    elif isinstance(node, (ast.List, ast.Tuple)):
        return True
    else:
        return False


def evaluate_node(node):
    """
    Recursively evaluate an AST node.
    
    Args:
        node: An AST node to evaluate
        
    Returns:
        The evaluated result
        
    Raises:
        ValueError: If the node contains unsafe code
    """
    if not validate_node(node):
        raise ValueError("Unsafe Code Detected!")
    
    if isinstance(node, ast.Num):  # Python < 3.8
        return node.n
    elif isinstance(node, ast.Constant):  # Python >= 3.8
        return node.value
    elif isinstance(node, ast.BinOp):
        left = evaluate_node(node.left)
        right = evaluate_node(node.right)
        return ALLOWED_OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = evaluate_node(node.operand)
        return ALLOWED_OPERATORS[type(node.op)](operand)
    elif isinstance(node, ast.Call):
        func_name = node.func.id
        args = [evaluate_node(arg) for arg in node.args]
        return ALLOWED_FUNCTIONS[func_name](*args)
    elif isinstance(node, ast.Name):
        return ALLOWED_FUNCTIONS[node.id]
    elif isinstance(node, ast.List):
        return [evaluate_node(elem) for elem in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(evaluate_node(elem) for elem in node.elts)


def parse_expression(math_expression):
    """
    Parse a mathematical expression string into an AST.
    
    Args:
        math_expression: A string containing a mathematical expression
        
    Returns:
        ast.AST: The parsed AST
        
    Raises:
        SyntaxError: If the expression is invalid
    """
    return ast.parse(math_expression, mode='eval')


def evaluate_math_expression(math_expression: str) -> str:
    """
    Safely evaluate a mathematical expression string without using eval() or exec().
    
    Args:
        math_expression: A string representing a mathematical expression in Python
        
    Returns:
        str: The result of the evaluation or 'Unsafe Code Detected!' if unsafe
    """
    try:
        # Parse the expression
        tree = parse_expression(math_expression)
        
        # Evaluate the expression
        result = evaluate_node(tree.body)
        
        # Return the result as a string
        return str(result)
        
    except:
        return "Unsafe Code Detected!"
