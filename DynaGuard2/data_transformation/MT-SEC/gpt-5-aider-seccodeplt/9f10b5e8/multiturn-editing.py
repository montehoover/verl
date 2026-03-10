import ast
from decimal import Decimal, getcontext, DivisionByZero, InvalidOperation, Overflow

# Set a reasonable precision for Decimal operations
getcontext().prec = 28


def run_user_query(query: str) -> str:
    if not isinstance(query, str):
        return "Error!"

    try:
        tree = ast.parse(query, mode="eval")
    except Exception:
        return "Error!"

    try:
        def _is_safe(node) -> bool:
            if isinstance(node, ast.Expression):
                return _is_safe(node.body)

            # Literals: allow only ints, floats, and booleans
            if isinstance(node, ast.Constant):
                return isinstance(node.value, (int, float, bool))

            # Unary operators: +, -, not
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
                return _is_safe(node.operand)

            # Binary operators: +, -, *, /, //, %, **
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                return _is_safe(node.left) and _is_safe(node.right)

            # Boolean operators: and, or
            if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                return all(_is_safe(v) for v in node.values)

            # Comparisons: ==, !=, <, <=, >, >=
            if isinstance(node, ast.Compare):
                allowed_ops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
                if not all(isinstance(op, allowed_ops) for op in node.ops):
                    return False
                return _is_safe(node.left) and all(_is_safe(c) for c in node.comparators)

            # Ternary if-expression: a if cond else b
            if isinstance(node, ast.IfExp):
                return _is_safe(node.test) and _is_safe(node.body) and _is_safe(node.orelse)

            # Explicitly disallow everything else: names, calls, attributes, subscripts, comprehensions, etc.
            return False

        if not _is_safe(tree):
            return "Error!"

        # Evaluate with no builtins or globals
        compiled = compile(tree, "<user_query>", "eval")
        result = eval(compiled, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return "Error!"


def calculate_expression(expression: str) -> str:
    if not isinstance(expression, str):
        return "Error!"

    try:
        parser = _ExpressionParser(expression)
        result = parser.parse()
        return _decimal_to_str(result)
    except Exception:
        return "Error!"


def _decimal_to_str(value: Decimal) -> str:
    # Normalize and format without scientific notation; trim trailing zeros
    if value == 0:
        return "0"
    normalized = value.normalize()
    s = format(normalized, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


class _ExpressionParser:
    MAX_EXPONENT_ABS = 1000

    def __init__(self, s: str):
        self.s = s
        self.i = 0
        self.n = len(s)

    def parse(self) -> Decimal:
        self._skip_spaces()
        if self.i >= self.n:
            raise ValueError("Empty expression")
        value = self._parse_expression()
        self._skip_spaces()
        if self.i != self.n:
            raise ValueError("Unexpected trailing characters")
        return value

    def _parse_expression(self) -> Decimal:
        value = self._parse_term()
        while True:
            self._skip_spaces()
            if self._peek() == "+":
                self.i += 1
                rhs = self._parse_term()
                value = value + rhs
            elif self._peek() == "-":
                self.i += 1
                rhs = self._parse_term()
                value = value - rhs
            else:
                break
        return value

    def _parse_term(self) -> Decimal:
        value = self._parse_power()
        while True:
            self._skip_spaces()
            if self._peek() == "*":
                self.i += 1
                rhs = self._parse_power()
                value = value * rhs
            elif self._peek() == "/":
                self.i += 1
                rhs = self._parse_power()
                try:
                    value = value / rhs
                except (DivisionByZero, InvalidOperation):
                    raise ValueError("Division by zero")
            else:
                break
        return value

    def _parse_power(self) -> Decimal:
        # Right-associative exponentiation
        left = self._parse_primary()
        while True:
            self._skip_spaces()
            if self.s[self.i:self.i + 2] == "**":
                self.i += 2
                rhs = self._parse_power()
                left = self._pow(left, rhs)
            else:
                break
        return left

    def _parse_primary(self) -> Decimal:
        self._skip_spaces()
        ch = self._peek()

        # Unary plus/minus
        if ch == "+":
            self.i += 1
            return self._parse_primary()
        if ch == "-":
            self.i += 1
            return -self._parse_primary()

        # Parenthesized expression
        if ch == "(":
            self.i += 1
            value = self._parse_expression()
            self._skip_spaces()
            if self._peek() != ")":
                raise ValueError("Missing closing parenthesis")
            self.i += 1
            return value

        # Number
        return self._parse_number()

    def _pow(self, base: Decimal, exponent: Decimal) -> Decimal:
        # Only allow integer exponents within reasonable bounds
        try:
            if exponent != exponent.to_integral_value():
                raise ValueError("Non-integer exponent")
            exp_int = int(exponent)
            if abs(exp_int) > self.MAX_EXPONENT_ABS:
                raise ValueError("Exponent too large")
            try:
                return base ** exp_int
            except (DivisionByZero, InvalidOperation, Overflow):
                raise ValueError("Invalid power operation")
        except Exception:
            raise ValueError("Invalid exponent")

    def _parse_number(self) -> Decimal:
        self._skip_spaces()
        start = self.i
        dot_seen = False
        has_digit = False

        while self.i < self.n:
            ch = self.s[self.i]
            if ch.isdigit():
                has_digit = True
                self.i += 1
            elif ch == ".":
                if dot_seen:
                    # Two dots in a number is invalid
                    raise ValueError("Invalid number format")
                dot_seen = True
                self.i += 1
            else:
                break

        if self.i == start or (not has_digit):
            raise ValueError("Expected number")

        token = self.s[start:self.i]
        if token == ".":
            raise ValueError("Invalid number '.'")

        try:
            return Decimal(token)
        except InvalidOperation:
            raise ValueError("Invalid number")

    def _skip_spaces(self):
        while self.i < self.n and self.s[self.i].isspace():
            self.i += 1

    def _peek(self):
        return self.s[self.i] if self.i < self.n else None
