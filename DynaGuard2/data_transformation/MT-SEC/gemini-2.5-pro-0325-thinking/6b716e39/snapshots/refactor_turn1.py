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


def evaluate_user_math(math_expr: str) -> str:
    """
    Evaluates a user-supplied mathematical or logical expression string safely.

    Args:
        math_expr: The string expression to evaluate.

    Returns:
        The result of the evaluation as a string, or 'Processing Failed!'
        if there's an error, or the expression is deemed unsafe.
    """
    if not isinstance(math_expr, str):
        return 'Processing Failed!'

    if len(math_expr) > 1000:  # Basic length guard
        return 'Processing Failed!'

    try:
        # Parse the expression. mode='eval' expects a single expression.
        # The result is an ast.Expression node.
        parsed_ast_expression = ast.parse(math_expr, mode='eval')

        # The actual expression content is in parsed_ast_expression.body
        # Validate the AST structure starting from the body of the expression.
        if not _is_safe_ast_node(parsed_ast_expression.body):
            return 'Processing Failed!'

        # If AST is safe, evaluate it.
        result = _evaluate_ast_node(parsed_ast_expression.body)
        
        # Ensure result is of an expected type (number or boolean)
        if not isinstance(result, (int, float, bool)):
            # This might happen if _evaluate_ast_node has a bug or unhandled safe type
            return 'Processing Failed!'
            
        return str(result)

    except SyntaxError:
        # Invalid Python syntax in the expression string
        return 'Processing Failed!'
    except ZeroDivisionError:
        # Specific arithmetic error
        return 'Processing Failed!'
    except OverflowError:
        # Result of an arithmetic operation is too large
        return 'Processing Failed!'
    except TypeError:
        # Type mismatch during an operation (e.g. if an op was misconfigured)
        # or if an unsupported node type slipped through evaluation logic.
        return 'Processing Failed!'
    except Exception:
        # Catch-all for any other unexpected errors during parsing or evaluation
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
