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
    """
    Recursively checks if an Abstract Syntax Tree (AST) node and all its children
    are safe for evaluation.

    Safety is determined by checking if the node type and any associated operators
    are present in predefined allowlists (_ALLOWED_NODES, _ALLOWED_BIN_OPS, etc.).
    For ast.Constant nodes, only numeric and boolean values are permitted.

    Args:
        node: The AST node to check.

    Returns:
        True if the node and its children are safe, False otherwise.
    """
    node_type = type(node)

    # Guard clause: Check if the node type itself is allowed.
    if node_type not in _ALLOWED_NODES:
        return False

    # Specific checks based on node type
    if isinstance(node, ast.Constant):
        # Allow numbers (int, float) and booleans. Disallow strings, None, etc.
        if not isinstance(node.value, (int, float, bool)):
            return False
    elif isinstance(node, ast.BinOp):
        # Check if the binary operator is allowed.
        if type(node.op) not in _ALLOWED_BIN_OPS:
            return False
    elif isinstance(node, ast.UnaryOp):
        # Check if the unary operator is allowed.
        if type(node.op) not in _ALLOWED_UNARY_OPS:
            return False
    elif isinstance(node, ast.Compare):
        # Check if all comparison operators in a chain are allowed.
        for op_type in node.ops:
            if type(op_type) not in _ALLOWED_COMP_OPS:
                return False
    elif isinstance(node, ast.BoolOp):
        # Check if the boolean operator (And, Or) is allowed.
        if type(node.op) not in _ALLOWED_BOOL_OPS:
            return False

    # Recursively check all child nodes.
    # If any child node is unsafe, this node is considered unsafe.
    for child_node in ast.iter_child_nodes(node):
        if not _is_safe_ast_node(child_node):
            return False
            
    return True # Node and all its children are safe.


def _evaluate_ast_node(node):
    """
    Recursively evaluates a pre-validated (safe) AST node.

    This function assumes `_is_safe_ast_node` has already vetted the node.
    It translates AST nodes into their corresponding Python operations or values.

    Args:
        node: The safe AST node to evaluate.

    Returns:
        The result of the evaluation (typically a number or boolean).

    Raises:
        TypeError: If an unsupported node type is encountered during evaluation
                   (should ideally be caught by `_is_safe_ast_node` beforehand).
        ZeroDivisionError: If a division by zero occurs.
        OverflowError: If an arithmetic operation results in a number too large to represent.
    """
    if isinstance(node, ast.Constant):
        # Base case: A constant node evaluates to its value.
        return node.value
    elif isinstance(node, ast.BinOp):
        # Binary operation: Evaluate left and right operands, then apply the operator.
        left = _evaluate_ast_node(node.left)
        right = _evaluate_ast_node(node.right)
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        # Unary operation: Evaluate the operand, then apply the operator.
        operand = _evaluate_ast_node(node.operand)
        return _ALLOWED_UNARY_OPS[type(node.op)](operand)
    elif isinstance(node, ast.Compare):
        # Comparison: Evaluate left, then chain comparisons with subsequent comparators.
        current_left_val = _evaluate_ast_node(node.left)
        for i, op_type in enumerate(node.ops):
            comparator_node = node.comparators[i]
            current_right_val = _evaluate_ast_node(comparator_node)
            op_func = _ALLOWED_COMP_OPS[type(op_type)]
            # If any comparison in the chain is false, the whole expression is false.
            if not op_func(current_left_val, current_right_val):
                return False  # Short-circuit evaluation
            current_left_val = current_right_val # For chained comparisons like 1 < x < 5
        return True  # All comparisons in the chain are true.
    elif isinstance(node, ast.BoolOp):
        # Boolean operation (And, Or): Evaluate all values.
        evaluated_values = [_evaluate_ast_node(v) for v in node.values]
        if type(node.op) is ast.And:
            return all(evaluated_values)
        elif type(node.op) is ast.Or:
            return any(evaluated_values)
    elif isinstance(node, ast.Expression):
        # An Expression node wraps the actual expression body. Evaluate the body.
        # This is typically the root node returned by ast.parse(..., mode='eval').
        return _evaluate_ast_node(node.body)
    else:
        # This case should ideally not be reached if _is_safe_ast_node is comprehensive
        # and has validated the structure correctly. It acts as a safeguard.
        raise TypeError(f"Unsupported node type for evaluation: {type(node)}")


def _parse_expression(math_expr: str) -> ast.AST:
    """
    Parses the input expression string into an Abstract Syntax Tree (AST)
    and performs initial validation on the input string itself.

    Args:
        math_expr: The string expression to parse.

    Returns:
        The body of the parsed AST (e.g., the actual expression node like ast.BinOp),
        not the top-level ast.Expression wrapper.

    Raises:
        ValueError: If `math_expr` is not a string or exceeds the maximum allowed length.
        SyntaxError: If `math_expr` is not a valid Python expression.
    """
    # Guard clause: Validate input type.
    if not isinstance(math_expr, str):
        raise ValueError("Input expression must be a string.")

    # Guard clause: Validate input length to prevent overly long expressions.
    if len(math_expr) > 1000:
        raise ValueError("Input expression is too long (max 1000 characters).")

    # Attempt to parse the expression.
    # ast.parse with mode='eval' expects a single expression.
    # It returns an ast.Expression node whose 'body' attribute contains the actual expression AST.
    # This can raise SyntaxError if the expression is malformed.
    parsed_ast_expression = ast.parse(math_expr, mode='eval')
    
    # Return the core expression node, not the ast.Expression wrapper.
    return parsed_ast_expression.body


def _security_check_ast(node: ast.AST) -> bool:
    """
    Performs a security validation on the provided AST node.

    This function delegates to `_is_safe_ast_node` to recursively check
    if all parts of the AST are within the defined allowlists.

    Args:
        node: The AST node (typically the body of an expression) to validate.

    Returns:
        True if the AST is deemed safe for evaluation, False otherwise.
    """
    # Validate the AST structure starting from the given node.
    # _is_safe_ast_node will recursively check all child nodes.
    return _is_safe_ast_node(node)


def _calculate_ast(node: ast.AST): # -> Union[int, float, bool]
    """
    Evaluates a pre-validated and security-checked AST node.

    This function delegates to `_evaluate_ast_node` for the recursive evaluation.
    It also performs a final check on the result type to ensure it's a number or boolean.

    Args:
        node: The safe AST node to evaluate.

    Returns:
        The result of the calculation (an int, float, or bool).

    Raises:
        ZeroDivisionError: If a division by zero occurs during evaluation.
        OverflowError: If an arithmetic operation results in a number too large.
        TypeError: If the evaluation results in an unexpected data type (not int, float, or bool),
                   or if an internal error occurs in `_evaluate_ast_node` with type handling.
    """
    # Evaluate the AST node.
    # _evaluate_ast_node can raise ZeroDivisionError, OverflowError, or TypeError (e.g., for unsupported ops
    # that might have slipped past _is_safe_ast_node, though unlikely).
    result = _evaluate_ast_node(node)
    
    # Guard clause: Ensure the final result is of an expected type (number or boolean).
    # This is a safeguard, as _evaluate_ast_node should only produce these types from allowed operations.
    if not isinstance(result, (int, float, bool)):
        # This indicates an unexpected state, possibly an issue with _ALLOWED_NODES/_OPS or _evaluate_ast_node logic.
        raise TypeError("Calculation resulted in an unexpected data type.")
            
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
