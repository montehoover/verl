from decimal import Decimal, getcontext, DivisionByZero, InvalidOperation, Overflow

# Set a reasonable precision for Decimal operations
getcontext().prec = 28


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
