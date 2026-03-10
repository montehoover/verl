import re
import ast
import operator

def substitute_variables(expression: str, variables: dict) -> str:
    """
    Substitute variable names in the given expression with their values from the variables dict.
    Only identifiers of the form [A-Za-z_]\\w* are considered variables.
    Identifiers not present in the dict are left unchanged.
    """
    if not expression:
        return expression
    if not variables:
        return expression

    pattern = re.compile(r"[A-Za-z_]\w*")

    def replace(match: re.Match) -> str:
        name = match.group(0)
        if name in variables:
            return str(variables[name])
        return name

    return pattern.sub(replace, expression)


def evaluate_expression(expression: str) -> str:
    """
    Evaluate a simple arithmetic expression containing numbers and +, -, *, / and parentheses.
    Returns the computed result as a string.
    """
    if expression is None:
        raise ValueError("expression must be a string")
    if expression.strip() == "":
        return expression

    bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in bin_ops:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            return bin_ops[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in unary_ops:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = _eval(node.operand)
            return unary_ops[op_type](operand)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported literal: {node.value!r}")
        # For compatibility with very old Python versions where numbers are ast.Num
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n  # type: ignore[union-attr]
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    tree = ast.parse(expression, mode="eval")
    result = _eval(tree)

    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)
