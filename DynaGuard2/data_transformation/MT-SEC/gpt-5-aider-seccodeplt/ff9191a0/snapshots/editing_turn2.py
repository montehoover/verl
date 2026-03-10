import ast
from typing import Union, Dict, Optional


def evaluate_simple_expression(expression: str, variables: Optional[Dict[str, Union[int, float]]] = None) -> str:
    """
    Evaluate a simple mathematical expression given as a string.
    Supports: addition (+), subtraction (-), multiplication (*), division (/), and parentheses.
    Variables can be provided via the 'variables' mapping (dict) where keys are variable names
    and values are numeric (int or float). All variables used in the expression must be present
    in the mapping.

    Returns the computed result as a string.
    Raises ValueError if the expression is invalid, a variable is missing, or computation fails.
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")

    if variables is None:
        variables = {}
    if not isinstance(variables, dict):
        raise ValueError("Variables must be a dictionary.")
    for k, v in variables.items():
        if not isinstance(k, str) or not k.isidentifier():
            raise ValueError("Variable names must be identifier strings.")
        # Disallow bools (since bool is a subclass of int) and require numeric values
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("Variable values must be numeric (int or float).")

    expr = expression.strip()
    if not expr:
        raise ValueError("Expression cannot be empty.")

    try:
        node = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError("Invalid expression.") from e

    def _eval(n: ast.AST) -> Union[int, float]:
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            op = n.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right
            raise ValueError("Unsupported operator.")

        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator.")

        if isinstance(n, ast.Name):
            name = n.id
            if name not in variables:
                raise ValueError(f"Missing variable: {name}")
            value = variables[name]
            # value type already validated
            return value  # type: ignore[return-value]

        # Python 3.8+: ast.Constant; older versions may use ast.Num
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Only numeric literals are allowed.")
        if hasattr(ast, "Num") and isinstance(n, getattr(ast, "Num")):
            return n.n  # type: ignore[attr-defined]

        # Disallow names, calls, attributes, subscripts, etc.
        raise ValueError("Invalid expression content.")

    try:
        result = _eval(node)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError("Computation failed.") from e

    return str(result)
