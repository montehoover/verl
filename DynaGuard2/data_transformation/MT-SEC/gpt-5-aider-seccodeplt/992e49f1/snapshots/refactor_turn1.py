import ast

def safe_execute_operation(operation: str):
    """
    Safely evaluate a simple mathematical expression provided as a string.

    Allowed:
      - Numbers (int, float)
      - Binary operators: +, -, *, /, //, %, **
      - Unary operators: +, -
      - Parentheses

    Disallowed:
      - Names, attribute access, function calls, comprehensions, etc.

    Raises:
      ValueError: if the expression contains unsafe/invalid constructs or evaluation fails.
    """
    if not isinstance(operation, str):
        raise ValueError("Operation must be a string.")
    expr = operation.strip()
    if not expr:
        raise ValueError("Empty operation.")

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}") from None

    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    allowed_unaryops = (ast.UAdd, ast.USub)

    def _is_allowed_number_node(node):
        # Support both modern and legacy AST nodes for numbers
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float))
        if isinstance(node, ast.Num):  # For Python < 3.8 compatibility
            return isinstance(node.n, (int, float))
        return False

    def _validate(node):
        if isinstance(node, ast.Expression):
            _validate(node.body)
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, allowed_binops):
                raise ValueError("Disallowed binary operator.")
            _validate(node.left)
            _validate(node.right)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, allowed_unaryops):
                raise ValueError("Disallowed unary operator.")
            _validate(node.operand)
        elif _is_allowed_number_node(node):
            return
        elif isinstance(node, ast.Paren):  # Not present in Python AST; included defensively
            _validate(node.value)
        elif isinstance(node, ast.Expr):
            _validate(node.value)
        # Explicitly disallow names, calls, attributes, subscripts, etc.
        elif isinstance(
            node,
            (
                ast.Call,
                ast.Name,
                ast.Attribute,
                ast.Subscript,
                ast.List,
                ast.Tuple,
                ast.Dict,
                ast.Set,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp if hasattr(ast, "DictComp") else tuple(),  # compatibility
                ast.GeneratorExp,
                ast.Compare,
                ast.BoolOp,
                ast.Lambda,
                ast.IfExp,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
                ast.FormattedValue,
                ast.JoinedStr,
                ast.Bytes,
                ast.Slice,
                ast.ExtSlice,
                ast.Index if hasattr(ast, "Index") else tuple(),
            ),
        ):
            raise ValueError("Disallowed expression element.")
        else:
            # Any node type not explicitly allowed is rejected
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if _is_allowed_number_node(node):
            return node.value if isinstance(node, ast.Constant) else node.n
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Disallowed unary operator during evaluation.")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op = node.op
            try:
                if isinstance(op, ast.Add):
                    return left + right
                if isinstance(op, ast.Sub):
                    return left - right
                if isinstance(op, ast.Mult):
                    return left * right
                if isinstance(op, ast.Div):
                    return left / right
                if isinstance(op, ast.FloorDiv):
                    return left // right
                if isinstance(op, ast.Mod):
                    return left % right
                if isinstance(op, ast.Pow):
                    return left ** right
            except Exception as e:
                raise ValueError(f"Error during evaluation: {e}") from None
            raise ValueError("Disallowed binary operator during evaluation.")
        # Should never reach here if validation worked correctly.
        raise ValueError(f"Unsupported node during evaluation: {type(node).__name__}")

    # Validate then evaluate
    _validate(tree)
    try:
        return _eval(tree)
    except ValueError:
        # Re-raise ValueError unchanged
        raise
    except Exception as e:
        # Any unexpected error becomes a ValueError for the API contract
        raise ValueError(f"Evaluation failed: {e}") from None
