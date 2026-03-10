import re
import operator
import ast

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Only allow typical ASCII characters for variables and arithmetic expressions
_ALLOWED_CHARS_PATTERN = re.compile(r'^[A-Za-z0-9_\s\+\-\*/\^\.\(\)]+$')


class _SafeEvaluator(ast.NodeVisitor):
    def __init__(self, vars_mapping):
        self.vars = vars_mapping

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)

        if op_type not in _ALLOWED_BINOPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")

        try:
            result = _ALLOWED_BINOPS[op_type](left, right)
        except ZeroDivisionError as e:
            raise ValueError("Division by zero") from e
        except OverflowError as e:
            raise ValueError("Numeric overflow during evaluation") from e
        except Exception as e:
            raise ValueError("Failed to evaluate expression") from e

        return result

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_type = type(node.op)

        if op_type not in _ALLOWED_UNARYOPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

        try:
            return _ALLOWED_UNARYOPS[op_type](operand)
        except Exception as e:
            raise ValueError("Failed to evaluate unary operation") from e

    def visit_Name(self, node):
        name = node.id
        if name not in self.vars:
            raise ValueError(f"Unknown variable: {name}")
        value = self.vars[name]
        if isinstance(value, bool):
            # Disallow booleans explicitly (bool is a subclass of int)
            raise ValueError(f"Invalid non-numeric value for variable '{name}'")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Invalid non-numeric value for variable '{name}'")
        return value

    def visit_Constant(self, node):
        value = node.value
        if isinstance(value, bool):
            raise ValueError("Boolean constants are not allowed")
        if not isinstance(value, (int, float)):
            raise ValueError("Only numeric constants are allowed")
        return value

    # Python <3.8 compatibility for numeric literals
    def visit_Num(self, node):  # pragma: no cover
        value = node.n
        if isinstance(value, bool):
            raise ValueError("Boolean constants are not allowed")
        if not isinstance(value, (int, float)):
            raise ValueError("Only numeric constants are allowed")
        return value

    # Explicitly forbid everything else
    def generic_visit(self, node):
        forbidden = (
            ast.Call,
            ast.Attribute,
            ast.Subscript,
            ast.List,
            ast.Tuple,
            ast.Dict,
            ast.Set,
            ast.Compare,
            ast.BoolOp,
            ast.IfExp,
            ast.Lambda,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp if hasattr(ast, "DictComp") else tuple(),  # compatibility
            ast.GeneratorExp,
            ast.AugAssign,
            ast.Assign,
            ast.Delete,
            ast.Yield,
            ast.YieldFrom,
            ast.Await,
            ast.With,
            ast.Raise,
            ast.Try,
            ast.For,
            ast.While,
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
            ast.Slice,
            ast.ExtSlice if hasattr(ast, "ExtSlice") else tuple(),
        )
        if isinstance(node, forbidden):
            raise ValueError(f"Unsupported or unsafe expression element: {type(node).__name__}")
        return super().generic_visit(node)


def _format_number(value):
    # Normalize ints
    if isinstance(value, int):
        return str(value)

    # Normalize floats
    if isinstance(value, float):
        # Check for NaN or Infinity without importing math
        if value != value:
            raise ValueError("Result is NaN")
        if value == float("inf") or value == float("-inf"):
            raise ValueError("Result is infinite")

        if value.is_integer():
            return str(int(value))
        # Use up to 15 significant digits and strip trailing zeros
        s = format(value, ".15g")
        return s

    # Fallback (shouldn't reach here with our evaluator)
    return str(value)


def simplify_math_expression(formula_str: str, vars_mapping: dict) -> str:
    """
    Evaluate and simplify a mathematical expression string with variables.

    Args:
        formula_str (str): A string representing a mathematical formula potentially containing variables.
        vars_mapping (dict): A mapping of variable names to their numeric (int or float) values.

    Returns:
        str: The result after computing the expression, returned as a string.

    Raises:
        ValueError: If an error occurs due to an invalid expression or unsuccessful processing.
    """
    if not isinstance(formula_str, str):
        raise ValueError("formula_str must be a string")
    if not isinstance(vars_mapping, dict):
        raise ValueError("vars_mapping must be a dictionary")

    expr = formula_str.strip()
    if not expr:
        raise ValueError("Empty expression")

    # Replace caret (^) with exponentiation (**) for user convenience
    if "^" in expr:
        expr = expr.replace("^", "**")

    # Basic character whitelist to prevent obviously unsafe content
    if not _ALLOWED_CHARS_PATTERN.fullmatch(expr):
        raise ValueError("Expression contains invalid characters")

    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError("Invalid expression syntax") from e

    evaluator = _SafeEvaluator(vars_mapping)

    try:
        result = evaluator.visit(parsed)
    except ValueError:
        # Re-raise our explicit ValueErrors untouched
        raise
    except Exception as e:
        raise ValueError("Failed to process expression") from e

    return _format_number(result)
