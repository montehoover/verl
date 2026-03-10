import ast
import operator as op

# Supported operators
ALLOWED_OPERATORS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg
}

# Allowed node types in the AST
ALLOWED_NODES = (
    ast.Expression, ast.Call, ast.Name, ast.Load,
    ast.BinOp, ast.UnaryOp, ast.Num, # ast.Num is deprecated in Python 3.8+, use ast.Constant
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub
)


def _eval_ast_node(node):
    """
    Recursively evaluate an AST node.
    Only allows nodes and operations deemed safe.
    """
    if isinstance(node, ast.Constant): # For numbers and potentially other constants
        if isinstance(node.value, (int, float)):
            return node.value
        else:
            raise TypeError("Unsupported constant type")
    elif isinstance(node, ast.Num): # For older Python versions (pre 3.8)
        return node.n
    elif isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        operator_func = ALLOWED_OPERATORS.get(type(node.op))
        if operator_func:
            return operator_func(left, right)
        else:
            raise TypeError(f"Unsupported binary operator: {type(node.op)}")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand)
        operator_func = ALLOWED_OPERATORS.get(type(node.op))
        if operator_func:
            return operator_func(operand)
        else:
            raise TypeError(f"Unsupported unary operator: {type(node.op)}")
    else:
        raise TypeError(f"Unsupported AST node type: {type(node)}")


def _is_safe_ast(node):
    """
    Check if all nodes in the AST are allowed.
    """
    if not isinstance(node, ALLOWED_NODES):
        return False
    for child_node in ast.iter_child_nodes(node):
        if not _is_safe_ast(child_node):
            return False
    return True


def evaluate_arith_expression(input_expr: str) -> str:
    """
    Interprets and executes basic arithmetic expressions given as input_expr.
    If the expression evaluates successfully, the result is returned as a string.
    If the evaluation fails or contains any potentially dangerous elements,
    returns 'Computation Error!'.
    """
    if not input_expr:
        return 'Computation Error!'

    try:
        # 1. Parse the expression into an Abstract Syntax Tree (AST)
        parsed_ast = ast.parse(input_expr, mode='eval')

        # 2. Validate the AST to ensure it only contains allowed nodes/operations
        if not _is_safe_ast(parsed_ast.body):
            return 'Computation Error!'

        # 3. Evaluate the AST
        # The _eval_ast_node function handles the actual computation based on the AST structure.
        # We pass parsed_ast.body because ast.parse with mode='eval' returns an ast.Expression node,
        # and its 'body' attribute contains the actual expression node (e.g., ast.BinOp, ast.Num).
        result = _eval_ast_node(parsed_ast.body)

        # Ensure the result is a number (int or float)
        if not isinstance(result, (int, float)):
            return 'Computation Error!' # Should be caught by _eval_ast_node, but as a safeguard

        return str(result)

    except (SyntaxError, TypeError, ZeroDivisionError, OverflowError, ValueError, RecursionError):
        # Catch common errors during parsing or evaluation
        return 'Computation Error!'
    except Exception:
        # Catch any other unexpected errors
        return 'Computation Error!'
