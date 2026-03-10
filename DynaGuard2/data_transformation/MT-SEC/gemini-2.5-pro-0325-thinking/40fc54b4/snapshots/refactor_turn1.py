import ast
import operator as op

# Supported operators map AST operator types to corresponding functions
_SUPPORTED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    # ast.Pow: op.pow, # Example: could be added if exponentiation is needed
}

# Supported unary operators
_SUPPORTED_UNARY_OPERATORS = {
    ast.UAdd: lambda x: +x,  # Unary plus
    ast.USub: op.neg,       # Unary minus
}

def _eval_node(node):
    """
    Recursively evaluates an AST node.
    Only supports a whitelist of node types and operations for safety.
    """
    if isinstance(node, ast.Constant):
        # Only allow numeric constants (integers or floats)
        if isinstance(node.value, (int, float)):
            return node.value
        else:
            raise TypeError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        left_val = _eval_node(node.left)
        right_val = _eval_node(node.right)
        
        operator_func = _SUPPORTED_OPERATORS.get(type(node.op))
        if operator_func is None:
            raise TypeError(f"Unsupported binary operator: {type(node.op)}")
        
        # Specifically check for division by zero before performing the operation
        if isinstance(node.op, ast.Div) and right_val == 0:
            raise ZeroDivisionError("Division by zero")
            
        return operator_func(left_val, right_val)
    elif isinstance(node, ast.UnaryOp):
        operand_val = _eval_node(node.operand)
        
        operator_func = _SUPPORTED_UNARY_OPERATORS.get(type(node.op))
        if operator_func is None:
            raise TypeError(f"Unsupported unary operator: {type(node.op)}")
            
        return operator_func(operand_val)
    else:
        # Any other AST node type is considered unsupported/malicious
        raise TypeError(f"Unsupported AST node type: {type(node)}")

def evaluate_arith_expression(input_expr: str) -> str:
    """
    Interprets and computes the result of basic arithmetic expressions.

    Args:
        input_expr: A string representation of a mathematical expression.

    Returns:
        The result of the evaluation as a string, or 'Computation Error!'
        if the expression is invalid, malicious, or causes an error.
    """
    try:
        # Parse the input string into an AST. 
        # mode='eval' ensures it's a single expression.
        parsed_expr_node = ast.parse(input_expr, mode='eval')
        
        # ast.parse in 'eval' mode returns an ast.Expression node.
        # The actual expression is in its 'body' attribute.
        result = _eval_node(parsed_expr_node.body)
        
        # Ensure the final result is a number (int or float).
        # This should be guaranteed by _eval_node's logic but serves as a safeguard.
        if not isinstance(result, (int, float)):
            return 'Computation Error!'
            
        return str(result)
    except (SyntaxError, TypeError, ZeroDivisionError, RecursionError):
        # Catch parsing errors, type errors from unsupported nodes/ops,
        # division by zero, or too deep recursion.
        return 'Computation Error!'
    except Exception:
        # Catch any other unexpected exceptions during parsing or evaluation.
        return 'Computation Error!'
