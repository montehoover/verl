import ast
import operator as op

# Allowed AST node types
_ALLOWED_NODES = {
    ast.Expression,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.BoolOp,
}

# Allowed operators for BinOp mapped to their functions
_ALLOWED_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

# Allowed operators for UnaryOp mapped to their functions
_ALLOWED_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
    ast.Not: op.not_,
}

# Allowed operators for Compare mapped to their functions
_ALLOWED_COMP_OPS = {
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Gt: op.gt,
    ast.GtE: op.ge,
}

# Allowed operators for BoolOp (custom handling in _eval_ast_node)
_ALLOWED_BOOL_OPS = {
    ast.And,
    ast.Or,
}


def _is_safe_ast_node(node):
    """Recursively check if an AST node and its children are safe."""
    node_type = type(node)

    if node_type not in _ALLOWED_NODES:
        return False

    if isinstance(node, ast.Constant):
        # Allow numbers (int, float) and booleans. Disallow strings, None, etc.
        if not isinstance(node.value, (int, float, bool)):
            return False
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BIN_OPS:
            return False
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARY_OPS:
            return False
    elif isinstance(node, ast.Compare):
        for op_type in node.ops:
            if type(op_type) not in _ALLOWED_COMP_OPS:
                return False
    elif isinstance(node, ast.BoolOp):
        if type(node.op) not in _ALLOWED_BOOL_OPS:
            return False

    for child_node in ast.iter_child_nodes(node):
        if not _is_safe_ast_node(child_node):
            return False
    return True


def _evaluate_ast_node(node):
    """Recursively evaluate a safe AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _evaluate_ast_node(node.left)
        right = _evaluate_ast_node(node.right)
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _evaluate_ast_node(node.operand)
        return _ALLOWED_UNARY_OPS[type(node.op)](operand)
    elif isinstance(node, ast.Compare):
        current_left_val = _evaluate_ast_node(node.left)
        for i, op_type in enumerate(node.ops):
            comparator_node = node.comparators[i]
            current_right_val = _evaluate_ast_node(comparator_node)
            op_func = _ALLOWED_COMP_OPS[type(op_type)]
            if not op_func(current_left_val, current_right_val):
                return False  # Short-circuit
            current_left_val = current_right_val
        return True  # All comparisons in the chain are true
    elif isinstance(node, ast.BoolOp):
        evaluated_values = [_evaluate_ast_node(v) for v in node.values]
        if type(node.op) is ast.And:
            return all(evaluated_values)
        elif type(node.op) is ast.Or:
            return any(evaluated_values)
    # This case should ideally not be reached if called on the body of an ast.Expression
    # and _is_safe_ast_node has validated the structure.
    # However, ast.Expression itself is a valid node type for the root.
    elif isinstance(node, ast.Expression):
        return _evaluate_ast_node(node.body)
    else:
        # Should be caught by _is_safe_ast_node, but as a safeguard:
        raise TypeError(f"Unsupported node type for evaluation: {type(node)}")


def _parse_expression(math_expr: str) -> ast.AST:
    """
    Parses the input expression string and performs initial validation.
    Raises ValueError for invalid input type or length, SyntaxError for parsing errors.
    Returns the body of the parsed AST (e.g., the actual expression node).
    """
    if not isinstance(math_expr, str):
        raise ValueError("Input expression must be a string.")

    if len(math_expr) > 1000:  # Basic length guard
        raise ValueError("Input expression is too long.")

    # Parse the expression. mode='eval' expects a single expression.
    # The result is an ast.Expression node.
    # SyntaxError can be raised here.
    parsed_ast_expression = ast.parse(math_expr, mode='eval')
    return parsed_ast_expression.body


def _security_check_ast(node: ast.AST) -> bool:
    """
    Performs security validation on the AST node.
    Returns True if the AST is safe, False otherwise.
    """
    # Validate the AST structure starting from the given node.
    return _is_safe_ast_node(node)


def _calculate_ast(node: ast.AST): # -> Union[int, float, bool]
    """
    Evaluates the validated AST node.
    Raises ZeroDivisionError, OverflowError, TypeError for calculation errors
    or if the result is not an expected type (int, float, bool).
    Returns the calculation result.
    """
    # If AST is safe, evaluate it.
    # _evaluate_ast_node can raise ZeroDivisionError, OverflowError, TypeError.
    result = _evaluate_ast_node(node)
    
    # Ensure result is of an expected type (number or boolean)
    if not isinstance(result, (int, float, bool)):
        # This might happen if _evaluate_ast_node has a bug or unhandled safe type
        raise TypeError("Calculation resulted in an unexpected type.")
            
    return result


def evaluate_user_math(math_expr: str) -> str:
    """
    Evaluates a user-supplied mathematical or logical expression string safely.

    Args:
        math_expr: The string expression to evaluate.

    Returns:
        The result of the evaluation as a string, or 'Processing Failed!'
        if there's an error, or the expression is deemed unsafe.
    """
    try:
        # Stage 1: Parse and Initial Validation
        # _parse_expression raises ValueError for bad input type/length, SyntaxError for parsing.
        ast_node_body = _parse_expression(math_expr)

        # Stage 2: Security Check
        # _security_check_ast returns False if not safe.
        if not _security_check_ast(ast_node_body):
            return 'Processing Failed!'  # Security violation

        # Stage 3: Calculation
        # _calculate_ast raises ZeroDivisionError, OverflowError, TypeError.
        result = _calculate_ast(ast_node_body)
            
        return str(result)

    # Catch specific errors explicitly raised or propagated by pipeline stages
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError):
        return 'Processing Failed!'
    # Catch-all for any other unexpected errors during any stage
    except Exception:
        # Ideally, log this unexpected error for debugging.
        return 'Processing Failed!'

if __name__ == '__main__':
    # Example Usage:
    print(f"'1 + 1': {evaluate_user_math('1 + 1')}")  # Expected: 2
    print(f"'2 * (3 + 4)': {evaluate_user_math('2 * (3 + 4)')}")  # Expected: 14
    print(f"'2 ** 3': {evaluate_user_math('2 ** 3')}")  # Expected: 8
    print(f"'10 / 2': {evaluate_user_math('10 / 2')}")  # Expected: 5.0
    print(f"'10 // 3': {evaluate_user_math('10 // 3')}") # Expected: 3
    print(f"'10 % 3': {evaluate_user_math('10 % 3')}")   # Expected: 1
    print(f"'-5': {evaluate_user_math('-5')}") # Expected: -5
    print(f"'+5': {evaluate_user_math('+5')}") # Expected: 5

    print(f"'1 < 2': {evaluate_user_math('1 < 2')}")  # Expected: True
    print(f"'1 > 2': {evaluate_user_math('1 > 2')}")  # Expected: False
    print(f"'1 == 1.0': {evaluate_user_math('1 == 1.0')}") # Expected: True
    print(f"'True and False': {evaluate_user_math('True and False')}") # Expected: False
    print(f"'True or False': {evaluate_user_math('True or False')}")   # Expected: True
    print(f"'not True': {evaluate_user_math('not True')}") # Expected: False
    print(f"'1 < 2 < 3': {evaluate_user_math('1 < 2 < 3')}") # Expected: True
    print(f"'1 < 3 < 2': {evaluate_user_math('1 < 3 < 2')}") # Expected: False

    # Error cases
    print(f"'1 / 0': {evaluate_user_math('1 / 0')}")  # Expected: Processing Failed!
    print(f"'import os': {evaluate_user_math('import os')}")  # Expected: Processing Failed! (SyntaxError)
    print(f"'().__class__': {evaluate_user_math('().__class__')}") # Expected: Processing Failed! (SyntaxError or Unsafe Node)
    print(f"""'eval("1+1")': {evaluate_user_math('eval("1+1")')}""") # Expected: Processing Failed! (Unsafe Node ast.Call)
    print(f"'x + 1': {evaluate_user_math('x + 1')}") # Expected: Processing Failed! (Unsafe Node ast.Name)
    print(f"""'\\\"hello\\\" + \\\"world\\\"': {evaluate_user_math('"hello" + "world"')}""") # Expected: Processing Failed! (ast.Constant string)
    print(f"Too long: {evaluate_user_math('1+' * 1000 + '1')}") # Expected: Processing Failed!
    print(f"None: {evaluate_user_math('None')}") # Expected: Processing Failed! (ast.Constant(None))
    print(f"lambda x: x: {evaluate_user_math('lambda x: x')}") # Expected: Processing Failed! (SyntaxError or Unsafe Node)
