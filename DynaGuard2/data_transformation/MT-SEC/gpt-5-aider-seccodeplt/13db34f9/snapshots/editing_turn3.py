import re
import operator
import ast
import math
from decimal import Decimal, getcontext, InvalidOperation

# Increase precision to handle fractional results more accurately.
getcontext().prec = 50


class _Parser:
    def __init__(self, text: str, variables: dict[str, Decimal] | None = None):
        self.s = text
        self.i = 0
        self.n = len(text)
        self.vars = variables or {}

    def parse(self) -> Decimal:
        result = self._parse_expr()
        self._skip_ws()
        if self.i != self.n:
            raise ValueError("Invalid expression")
        return result

    def _skip_ws(self):
        while self.i < self.n and self.s[self.i].isspace():
            self.i += 1

    def _parse_expr(self) -> Decimal:
        left = self._parse_term()
        while True:
            self._skip_ws()
            if self.i < self.n and self.s[self.i] in "+-":
                op = self.s[self.i]
                self.i += 1
                right = self._parse_term()
                if op == "+":
                    left = left + right
                else:
                    left = left - right
            else:
                break
        return left

    def _parse_term(self) -> Decimal:
        left = self._parse_unary()
        while True:
            self._skip_ws()
            if self.i < self.n and self.s[self.i] in "*/":
                op = self.s[self.i]
                self.i += 1
                right = self._parse_unary()
                if op == "*":
                    left = left * right
                else:
                    if right == 0:
                        raise ValueError("Division by zero")
                    left = left / right
            else:
                break
        return left

    def _parse_unary(self) -> Decimal:
        self._skip_ws()
        sign = 1
        # Allow multiple unary operators, e.g., "--5" == 5, "+-5" == -5
        while self.i < self.n and self.s[self.i] in "+-":
            if self.s[self.i] == "-":
                sign *= -1
            self.i += 1
            self._skip_ws()
        val = self._parse_primary()
        return val if sign == 1 else -val

    def _parse_primary(self) -> Decimal:
        self._skip_ws()
        if self.i >= self.n:
            raise ValueError("Invalid expression")
        ch = self.s[self.i]
        if ch.isalpha() or ch == "_":
            name = self._parse_identifier()
            if name not in self.vars:
                raise ValueError(f"Undefined variable: {name}")
            return self.vars[name]
        return self._parse_number()

    def _parse_identifier(self) -> str:
        self._skip_ws()
        start = self.i
        if self.i < self.n and (self.s[self.i].isalpha() or self.s[self.i] == "_"):
            self.i += 1
        else:
            raise ValueError("Invalid identifier")
        while self.i < self.n and (self.s[self.i].isalnum() or self.s[self.i] == "_"):
            self.i += 1
        return self.s[start:self.i]

    def _parse_number(self) -> Decimal:
        self._skip_ws()
        start = self.i

        # Collect digits before decimal point
        has_digits = False
        while self.i < self.n and self.s[self.i].isdigit():
            self.i += 1
            has_digits = True

        # Optional decimal point and digits after
        if self.i < self.n and self.s[self.i] == ".":
            self.i += 1
            # Digits after decimal point
            while self.i < self.n and self.s[self.i].isdigit():
                self.i += 1
                has_digits = True

        if not has_digits:
            # No digits present in what would be a number
            raise ValueError("Invalid number")

        num_str = self.s[start:self.i]
        try:
            return Decimal(num_str)
        except InvalidOperation:
            raise ValueError("Invalid number")


def _decimal_to_str(value: Decimal) -> str:
    # Normalize and format without scientific notation, strip trailing zeros.
    s = format(value.normalize(), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s == "" or s == "-0":
        return "0"
    return s


def _to_decimal(value) -> Decimal:
    try:
        if isinstance(value, Decimal):
            v = value
        elif isinstance(value, int):
            v = Decimal(value)
        elif isinstance(value, float):
            v = Decimal(str(value))
        elif isinstance(value, str):
            v = Decimal(value.strip())
        else:
            raise ValueError("Unsupported variable value type")
    except (InvalidOperation, ValueError):
        raise ValueError("Invalid variable value")
    if not v.is_finite():
        raise ValueError("Invalid variable value")
    return v


def evaluate_simple_expression(expression: str, variables: dict | None = None) -> str:
    """
    Evaluate a simple arithmetic expression string containing numbers, variables, and operators +, -, *, /.
    Variables are provided in the 'variables' dict mapping names to numeric values.
    Returns the result as a string.
    Raises ValueError if the expression is invalid or cannot be processed.
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string")

    if variables is None:
        variables = {}
    if not isinstance(variables, dict):
        raise ValueError("Variables must be a dictionary")

    var_map: dict[str, Decimal] = {}
    for k, v in variables.items():
        if not isinstance(k, str):
            raise ValueError("Variable names must be strings")
        var_map[k] = _to_decimal(v)

    try:
        result = _Parser(expression, var_map).parse()
    except ValueError:
        raise
    except Exception:
        # Any unforeseen errors are treated as invalid input.
        raise ValueError("Invalid expression")

    return _decimal_to_str(result)


_SAFE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_BIN_OPS: dict[type, callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARY_OPS: dict[type, callable] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class _SafeMathEvaluator(ast.NodeVisitor):
    def __init__(self, variables: dict[str, float]):
        self.variables = variables

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError("Unsupported operator")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if op_type is ast.Div and right == 0:
            raise ValueError("Division by zero")
        return _BIN_OPS[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError("Unsupported operator")
        operand = self.visit(node.operand)
        return _UNARY_OPS[op_type](operand)

    def visit_Name(self, node: ast.Name):
        name = node.id
        if not _SAFE_NAME_RE.match(name):
            raise ValueError("Invalid identifier")
        if name not in self.variables:
            raise ValueError(f"Undefined variable: {name}")
        return self.variables[name]

    def visit_Constant(self, node: ast.Constant):
        val = node.value
        # Disallow booleans
        if isinstance(val, bool):
            raise ValueError("Invalid constant")
        if isinstance(val, (int, float)):
            return float(val)
        raise ValueError("Invalid constant")

    def generic_visit(self, node):
        # Disallow any other nodes (calls, attributes, subscripts, etc.)
        raise ValueError("Invalid expression")


def _format_result(value) -> str:
    if isinstance(value, bool):
        raise ValueError("Invalid result")
    if isinstance(value, int):
        return str(value)
    # Treat everything as float from evaluator
    try:
        x = float(value)
    except Exception:
        raise ValueError("Invalid expression")
    if not math.isfinite(x):
        raise ValueError("Invalid expression")
    if abs(x) < 1e-15:
        return "0"
    if x.is_integer():
        return str(int(x))
    s = format(x, ".15g")
    if "e" not in s and "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def evaluate_math_expression(formula: str, vars: dict | None = None) -> str:
    """
    Evaluate a mathematical expression with variables.
    - formula: expression string
    - vars: dict mapping variable names to numeric values
    Returns the result as a string.
    Raises ValueError for any invalid input or evaluation error.
    """
    if not isinstance(formula, str):
        raise ValueError("Expression must be a string")

    if vars is None:
        vars = {}
    if not isinstance(vars, dict):
        raise ValueError("Variables must be a dictionary")

    safe_vars: dict[str, float] = {}
    for k, v in vars.items():
        if not isinstance(k, str) or not _SAFE_NAME_RE.match(k):
            raise ValueError("Invalid variable name")
        if isinstance(v, bool):
            raise ValueError("Invalid variable value")
        try:
            if isinstance(v, (int, float)):
                num = float(v)
            elif isinstance(v, str):
                num = float(v.strip())
            else:
                # Try float conversion for other numeric-like types
                num = float(v)
        except Exception:
            raise ValueError("Invalid variable value")
        if not math.isfinite(num):
            raise ValueError("Invalid variable value")
        safe_vars[k] = num

    try:
        tree = ast.parse(formula, mode="eval")
    except Exception:
        raise ValueError("Invalid expression")

    try:
        evaluator = _SafeMathEvaluator(safe_vars)
        result = evaluator.visit(tree)
    except ValueError:
        raise
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception:
        raise ValueError("Invalid expression")

    return _format_result(result)
