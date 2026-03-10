import ast
from typing import Any


def calculate_user_input(input_expr: str) -> str:
    """
    Safely evaluate a user-supplied arithmetic or logical expression.

    Args:
        input_expr: A string representing the expression to evaluate.

    Returns:
        str: The result of the evaluated expression as a string, or
             'Processing Failed!' if evaluation fails or the input is unsafe.
    """
    try:
        # Basic validation
        if not isinstance(input_expr, str):
            return 'Processing Failed!'
        expr = input_expr.strip()
        if not expr:
            return 'Processing Failed!'
        if len(expr) > 1000:
            return 'Processing Failed!'

        # Parse expression into AST
        try:
            tree = ast.parse(expr, mode='eval')
        except Exception:
            return 'Processing Failed!'

        # Allowlist of AST node types
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.BoolOp,
            ast.Compare,
            ast.Name,
            ast.Load,
            ast.Constant,
            # Operators
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
            ast.UAdd, ast.USub,
            ast.And, ast.Or,
            ast.Not,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        )

        # Ensure only allowed nodes are present and names are safe
        node_count = 0
        for node in ast.walk(tree):
            node_count += 1
            if node_count > 1000:
                return 'Processing Failed!'
            if not isinstance(node, allowed_nodes):
                return 'Processing Failed!'
            if isinstance(node, ast.Name):
                if node.id not in ('True', 'False'):
                    return 'Processing Failed!'

        # Evaluator
        def eval_node(node: ast.AST, depth: int = 0) -> Any:
            if depth > 50:
                raise ValueError('Expression too deep')

            if isinstance(node, ast.Expression):
                return eval_node(node.body, depth + 1)

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (bool, int, float)):
                    return node.value
                raise ValueError('Disallowed constant')

            if isinstance(node, ast.Name):
                if node.id == 'True':
                    return True
                if node.id == 'False':
                    return False
                raise ValueError('Disallowed name')

            if isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand, depth + 1)
                if isinstance(node.op, ast.Not):
                    if isinstance(operand, bool):
                        return not operand
                    raise ValueError('not expects boolean')
                if isinstance(node.op, (ast.UAdd, ast.USub)):
                    if isinstance(operand, (int, float)) and not isinstance(operand, bool):
                        return +operand if isinstance(node.op, ast.UAdd) else -operand
                    raise ValueError('Unary op expects number')
                raise ValueError('Disallowed unary operator')

            if isinstance(node, ast.BinOp):
                left = eval_node(node.left, depth + 1)
                right = eval_node(node.right, depth + 1)

                # Disallow booleans in arithmetic
                if (isinstance(left, bool) or isinstance(right, bool) or
                        not isinstance(left, (int, float)) or
                        not isinstance(right, (int, float))):
                    raise ValueError('Arithmetic expects numbers')

                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right
                if isinstance(node.op, ast.Mod):
                    return left % right
                # Disallow power and bitwise ops
                raise ValueError('Disallowed binary operator')

            if isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    result = True
                    for value_node in node.values:
                        val = eval_node(value_node, depth + 1)
                        if not isinstance(val, bool):
                            raise ValueError('and/or expects booleans')
                        result = result and val
                    return result
                if isinstance(node.op, ast.Or):
                    result = False
                    for value_node in node.values:
                        val = eval_node(value_node, depth + 1)
                        if not isinstance(val, bool):
                            raise ValueError('and/or expects booleans')
                        result = result or val
                    return result
                raise ValueError('Disallowed boolean operator')

            if isinstance(node, ast.Compare):
                left = eval_node(node.left, depth + 1)
                overall = True
                prev = left
                for op, comp_node in zip(node.ops, node.comparators):
                    right = eval_node(comp_node, depth + 1)

                    # Type constraints: compare numbers with numbers, bools with bools
                    if isinstance(prev, bool) or isinstance(right, bool):
                        if not (isinstance(prev, bool) and isinstance(right, bool)):
                            raise ValueError('Invalid comparison types')
                    else:
                        # Ensure numeric comparison
                        if not (isinstance(prev, (int, float)) and isinstance(right, (int, float))):
                            raise ValueError('Invalid comparison types')

                    if isinstance(op, ast.Eq):
                        ok = prev == right
                    elif isinstance(op, ast.NotEq):
                        ok = prev != right
                    elif isinstance(op, ast.Lt):
                        ok = prev < right
                    elif isinstance(op, ast.LtE):
                        ok = prev <= right
                    elif isinstance(op, ast.Gt):
                        ok = prev > right
                    elif isinstance(op, ast.GtE):
                        ok = prev >= right
                    else:
                        raise ValueError('Disallowed comparison operator')

                    if not ok:
                        overall = False
                    prev = right
                return overall

            # Any other node is disallowed
            raise ValueError(f'Disallowed syntax: {type(node).__name__}')

        result = eval_node(tree)
        return str(result)
    except Exception:
        return 'Processing Failed!'
