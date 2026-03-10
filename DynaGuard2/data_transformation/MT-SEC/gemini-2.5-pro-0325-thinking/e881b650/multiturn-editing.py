import re
import ast
import operator as op

# Supported operations
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

# Allowed node types in the AST
ALLOWED_NODES = (
    ast.Expression,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Num,  # Deprecated in Python 3.8, use ast.Constant
    ast.Constant, # For numbers and None
    ast.BinOp,
    ast.UnaryOp,
    ast.operator, # Superclass for operators
    ast.unaryop,  # Superclass for unary operators
    ast.cmpop,    # Superclass for comparison operators
    ast.boolop,   # Superclass for boolean operators
    ast.expr_context, # Superclass for expression context
)


def _eval_node(node):
    """
    Recursively evaluate an AST node.
    Only allows whitelisted operations and node types.
    """
    if isinstance(node, ast.Constant): # For Python 3.8+ for numbers
        if isinstance(node.value, (int, float)):
            return node.value
        else:
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.Num):  # For older Python versions (numbers)
        return node.n
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        operator_func = ALLOWED_OPERATORS.get(type(node.op))
        if operator_func:
            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("Division by zero.")
            return operator_func(left, right)
        else:
            raise ValueError(f"Unsupported binary operator: {type(node.op)}")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        operator_func = ALLOWED_OPERATORS.get(type(node.op))
        if operator_func:
            return operator_func(operand)
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op)}")
    else:
        # Check if the node type is in the allowed list or a subclass of allowed types
        if not any(isinstance(node, allowed_type) for allowed_type in ALLOWED_NODES):
            raise ValueError(f"Unsupported AST node type: {type(node)}")
        # If it's an allowed structural node (like Expression), recurse if it has a body/value
        if hasattr(node, 'body'): # e.g. ast.Expression
             return _eval_node(node.body)
        elif hasattr(node, 'value'): # e.g. ast.Attribute (though not used here)
             return _eval_node(node.value)

    raise ValueError(f"Unsupported expression component: {type(node)}")


def compute_expression(expr: str) -> float:
    """
    Evaluates a mathematical expression string safely.

    Supports basic arithmetic operations: +, -, *, /, ** (power),
    and unary + and -. Parentheses for grouping are also supported.

    Args:
        expr: The mathematical expression string to evaluate.
              Example: "2 * (3 + 4) - 5 / 2"

    Returns:
        The result of the evaluation as a float.

    Raises:
        ValueError: If the expression contains unsupported characters,
                    operations, is malformed, or attempts division by zero.
    """
    # Validate input string for allowed characters
    # Allows numbers, operators (+, -, *, /, .), parentheses, and whitespace.
    # Also allows 'e' or 'E' for scientific notation.
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s\^eE]+", expr):
        raise ValueError("Expression contains unsupported characters.")

    # Replace ^ with ** for Python's power operator if present
    expr = expr.replace('^', '**')

    try:
        # Parse the expression into an Abstract Syntax Tree (AST)
        # ast.parse returns a Module node, we need the Expression node from its body
        parsed_ast = ast.parse(expr, mode='eval')

        # Validate all nodes in the AST
        for node in ast.walk(parsed_ast):
            if not isinstance(node, ALLOWED_NODES) and \
               not type(node) in ALLOWED_OPERATORS:
                # Additional check for specific operator types if not covered by ALLOWED_NODES
                if not any(isinstance(node, op_type) for op_type in ALLOWED_OPERATORS.keys()):
                     raise ValueError(f"Unsupported language construct or operator: {type(node)}")

        # Evaluate the AST
        result = _eval_node(parsed_ast.body) # ast.parse with mode='eval' gives Expression, body is the actual expr
        if not isinstance(result, (int, float)):
            raise ValueError("Expression did not evaluate to a number.")
        return float(result)

    except SyntaxError:
        raise ValueError("Invalid syntax in expression.")
    except RecursionError: # AST too deep
        raise ValueError("Expression too complex or deeply nested.")
    except Exception as e:
        # Catch any other unexpected errors during parsing or evaluation
        # Re-raise as ValueError to maintain consistent error type for the caller
        if isinstance(e, ValueError): # if it's already a ValueError from _eval_node
            raise
        raise ValueError(f"Error evaluating expression: {e}")
