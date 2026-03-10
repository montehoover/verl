import ast
import operator
import math
import logging

# Configure logging
logger = logging.getLogger(__name__)

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
    node_type = type(node).__name__
    
    if isinstance(node, (ast.Num, ast.Constant)):
        if isinstance(node, ast.Constant):
            is_safe = isinstance(node.value, (int, float))
            if not is_safe:
                logger.warning(f"Unsafe constant value type: {type(node.value).__name__}")
            return is_safe
        return True
    elif isinstance(node, ast.BinOp):
        is_safe = type(node.op) in ALLOWED_OPERATORS
        if not is_safe:
            logger.warning(f"Disallowed binary operator: {type(node.op).__name__}")
        return is_safe
    elif isinstance(node, ast.UnaryOp):
        is_safe = type(node.op) in ALLOWED_OPERATORS
        if not is_safe:
            logger.warning(f"Disallowed unary operator: {type(node.op).__name__}")
        return is_safe
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            is_safe = node.func.id in ALLOWED_FUNCTIONS
            if not is_safe:
                logger.warning(f"Disallowed function call: {node.func.id}")
            return is_safe
        else:
            logger.warning(f"Complex function call not allowed: {type(node.func).__name__}")
            return False
    elif isinstance(node, ast.Name):
        is_safe = (node.id in ALLOWED_FUNCTIONS and 
                  isinstance(ALLOWED_FUNCTIONS[node.id], (int, float)))
        if not is_safe:
            logger.warning(f"Disallowed name reference: {node.id}")
        return is_safe
    elif isinstance(node, (ast.List, ast.Tuple)):
        return True
    else:
        logger.warning(f"Disallowed node type: {node_type}")
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
    
    node_type = type(node).__name__
    logger.debug(f"Evaluating node type: {node_type}")
    
    if isinstance(node, ast.Num):  # Python < 3.8
        result = node.n
        logger.debug(f"Evaluated number: {result}")
        return result
    elif isinstance(node, ast.Constant):  # Python >= 3.8
        result = node.value
        logger.debug(f"Evaluated constant: {result}")
        return result
    elif isinstance(node, ast.BinOp):
        left = evaluate_node(node.left)
        right = evaluate_node(node.right)
        op_name = type(node.op).__name__
        result = ALLOWED_OPERATORS[type(node.op)](left, right)
        logger.debug(f"Evaluated binary operation: {left} {op_name} {right} = {result}")
        return result
    elif isinstance(node, ast.UnaryOp):
        operand = evaluate_node(node.operand)
        op_name = type(node.op).__name__
        result = ALLOWED_OPERATORS[type(node.op)](operand)
        logger.debug(f"Evaluated unary operation: {op_name} {operand} = {result}")
        return result
    elif isinstance(node, ast.Call):
        func_name = node.func.id
        args = [evaluate_node(arg) for arg in node.args]
        result = ALLOWED_FUNCTIONS[func_name](*args)
        logger.debug(f"Evaluated function call: {func_name}({args}) = {result}")
        return result
    elif isinstance(node, ast.Name):
        result = ALLOWED_FUNCTIONS[node.id]
        logger.debug(f"Evaluated name reference: {node.id} = {result}")
        return result
    elif isinstance(node, ast.List):
        result = [evaluate_node(elem) for elem in node.elts]
        logger.debug(f"Evaluated list: {result}")
        return result
    elif isinstance(node, ast.Tuple):
        result = tuple(evaluate_node(elem) for elem in node.elts)
        logger.debug(f"Evaluated tuple: {result}")
        return result


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
    logger.debug(f"Parsing expression: {math_expression}")
    tree = ast.parse(math_expression, mode='eval')
    logger.debug("Expression parsed successfully")
    return tree


def evaluate_math_expression(math_expression: str) -> str:
    """
    Safely evaluate a mathematical expression string without using eval() or exec().
    
    Args:
        math_expression: A string representing a mathematical expression in Python
        
    Returns:
        str: The result of the evaluation or 'Unsafe Code Detected!' if unsafe
    """
    logger.info(f"Starting evaluation of expression: {math_expression}")
    
    try:
        # Parse the expression
        tree = parse_expression(math_expression)
        
        # Evaluate the expression
        result = evaluate_node(tree.body)
        
        # Return the result as a string
        result_str = str(result)
        logger.info(f"Evaluation successful. Result: {result_str}")
        return result_str
        
    except ValueError as e:
        logger.error(f"Unsafe code detected in expression: {math_expression}")
        return "Unsafe Code Detected!"
    except SyntaxError as e:
        logger.error(f"Syntax error in expression: {math_expression}. Error: {e}")
        return "Unsafe Code Detected!"
    except Exception as e:
        logger.error(f"Unexpected error evaluating expression: {math_expression}. Error: {type(e).__name__}: {e}")
        return "Unsafe Code Detected!"
