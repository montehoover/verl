import ast
import operator

def get_allowed_operators():
    """
    Return a dictionary of allowed AST operators mapped to their corresponding functions.
    
    Returns:
        dict: Mapping of AST operator types to operator functions
    """
    return {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv
    }

def validate_and_parse_expression(input_expr: str):
    """
    Parse and validate the input expression.
    
    Args:
        input_expr: String representation of a mathematical expression
        
    Returns:
        ast.AST: Parsed AST tree body
        
    Raises:
        SyntaxError: If the expression has invalid syntax
    """
    tree = ast.parse(input_expr, mode='eval')
    return tree.body

def evaluate_ast_node(node, allowed_operators):
    """
    Recursively evaluate an AST node safely.
    
    Args:
        node: AST node to evaluate
        allowed_operators: Dictionary of allowed operators
        
    Returns:
        int or float: Result of evaluation
        
    Raises:
        ValueError: If disallowed operations or types are encountered
    """
    if isinstance(node, ast.Constant):
        # Handle numeric constants
        if isinstance(node.value, (int, float)):
            return node.value
        else:
            raise ValueError("Invalid constant type")
            
    elif isinstance(node, ast.Num):  # For Python < 3.8 compatibility
        return node.n
        
    elif isinstance(node, ast.BinOp):
        # Binary operations
        if type(node.op) not in allowed_operators:
            raise ValueError("Disallowed operator")
        left = evaluate_ast_node(node.left, allowed_operators)
        right = evaluate_ast_node(node.right, allowed_operators)
        return allowed_operators[type(node.op)](left, right)
        
    elif isinstance(node, ast.UnaryOp):
        # Unary operations
        if type(node.op) not in allowed_operators:
            raise ValueError("Disallowed operator")
        operand = evaluate_ast_node(node.operand, allowed_operators)
        return allowed_operators[type(node.op)](operand)
        
    else:
        # Any other node type is not allowed
        raise ValueError("Disallowed expression type")

def evaluate_arith_expression(input_expr: str) -> str:
    """
    Safely evaluate basic arithmetic expressions.
    
    Args:
        input_expr: String representation of a mathematical expression
        
    Returns:
        str: Result of evaluation or 'Computation Error!' if evaluation fails
    """
    try:
        # Get allowed operators
        allowed_operators = get_allowed_operators()
        
        # Parse and validate the expression
        ast_body = validate_and_parse_expression(input_expr)
        
        # Evaluate the expression safely
        result = evaluate_ast_node(ast_body, allowed_operators)
        
        # Return result as string
        return str(result)
        
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, 
            AttributeError, KeyError, RecursionError):
        return "Computation Error!"
    except Exception:
        return "Computation Error!"
