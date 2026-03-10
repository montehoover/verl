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
