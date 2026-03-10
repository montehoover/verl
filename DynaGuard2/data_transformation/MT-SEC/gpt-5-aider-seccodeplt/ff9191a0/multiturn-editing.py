import re
import operator
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


def evaluate_expression_safely(math_expression: str, variable_mapping: Dict[str, Union[int, float]]) -> str:
    """
    Securely evaluate a mathematical expression with variables.

    Args:
        math_expression: The expression string to evaluate.
        variable_mapping: Dict mapping variable names (identifiers) to numeric values (int or float).

    Returns:
        The result of the expression as a string.

    Raises:
        ValueError: If the expression is invalid, contains unsupported constructs,
                    references missing variables, or computation fails.
    """
    if not isinstance(math_expression, str):
        raise ValueError("Expression must be a string.")

    if not isinstance(variable_mapping, dict):
        raise ValueError("Variables must be provided as a dictionary.")

    # Validate variables: names must be identifiers, values must be numeric (int or float), and not bool.
    for name, value in variable_mapping.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("Variable names must be identifier strings.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Variable values must be numeric (int or float).")

    expr = math_expression.strip()
    if not expr:
        raise ValueError("Expression cannot be empty.")

    # Operator maps for supported operations
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

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError("Invalid expression.") from e

    def _eval(node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in bin_ops:
                return bin_ops[op_type](left, right)
            raise ValueError("Unsupported operator.")

        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type in unary_ops:
                return unary_ops[op_type](operand)
            raise ValueError("Unsupported unary operator.")

        if isinstance(node, ast.Name):
            name = node.id
            if name not in variable_mapping:
                raise ValueError(f"Missing variable: {name}")
            return variable_mapping[name]

        # Numeric literals
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, bool):
                raise ValueError("Booleans are not allowed.")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Only numeric literals are allowed.")

        if hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):
            return node.n  # type: ignore[attr-defined]

        # Disallow everything else: calls, attributes, subscripts, comprehensions, etc.
        raise ValueError("Invalid expression content.")

    try:
        result = _eval(tree)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError("Computation failed.") from e

    return str(result)
