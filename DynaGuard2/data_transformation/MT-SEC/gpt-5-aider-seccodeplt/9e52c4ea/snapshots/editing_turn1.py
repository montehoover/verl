from decimal import Decimal, localcontext, DivisionByZero, InvalidOperation, Overflow, Underflow, Clamped


class _ParseError(Exception):
    pass


def _tokenize(expr: str):
    tokens = []
    i = 0
    n = len(expr)

    while i < n:
        ch = expr[i]

        if ch.isspace():
            i += 1
            continue

        if ch in '+-*/()':
            tokens.append(ch)
            i += 1
            continue

        if ch.isdigit() or ch == '.':
            start = i
            dot_count = 0
            while i < n and (expr[i].isdigit() or expr[i] == '.'):
                if expr[i] == '.':
                    dot_count += 1
                    if dot_count > 1:
                        raise _ParseError("Too many decimal points")
                i += 1
            num_str = expr[start:i]
            # Reject a standalone decimal point
            if num_str == '.' or num_str == '':
                raise _ParseError("Invalid number")
            try:
                tokens.append(Decimal(num_str))
            except InvalidOperation:
                raise _ParseError("Invalid number")
            continue

        # Any other character is invalid
        raise _ParseError(f"Invalid character: {ch}")

    return tokens


class _Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def _peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self, expected=None):
        tok = self._peek()
        if tok is None:
            raise _ParseError("Unexpected end of expression")
        if expected is not None and tok != expected:
            raise _ParseError(f"Expected '{expected}'")
        self.pos += 1
        return tok

    def parse(self):
        if not self.tokens:
            raise _ParseError("Empty expression")
        value = self._parse_expr()
        if self._peek() is not None:
            raise _ParseError("Unexpected token")
        return value

    def _parse_expr(self):
        value = self._parse_term()
        while True:
            tok = self._peek()
            if tok == '+':
                self._consume('+')
                rhs = self._parse_term()
                value = value + rhs
            elif tok == '-':
                self._consume('-')
                rhs = self._parse_term()
                value = value - rhs
            else:
                break
        return value

    def _parse_term(self):
        value = self._parse_factor()
        while True:
            tok = self._peek()
            if tok == '*':
                self._consume('*')
                rhs = self._parse_factor()
                value = value * rhs
            elif tok == '/':
                self._consume('/')
                rhs = self._parse_factor()
                value = value / rhs
            else:
                break
        return value

    def _parse_factor(self):
        tok = self._peek()

        if tok == '+':
            self._consume('+')
            return self._parse_factor()

        if tok == '-':
            self._consume('-')
            return -self._parse_factor()

        if tok == '(':
            self._consume('(')
            value = self._parse_expr()
            self._consume(')')
            return value

        if isinstance(tok, Decimal):
            self._consume()
            return tok

        raise _ParseError("Expected number or parenthesis")


def _format_decimal(d: Decimal) -> str:
    # Normalize to remove exponent/trailing zeros, keep fixed-point format
    s = format(d.normalize(), 'f')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    if s == '-0':
        s = '0'
    return s if s != '' else '0'


def evaluate_expression(expression: str) -> str:
    """
    Evaluate a simple arithmetic expression string and return the result as a string.
    Supports +, -, *, /, parentheses, and unary +/-.
    Returns 'Invalid Expression' on any error.
    """
    if not isinstance(expression, str):
        return 'Invalid Expression'

    try:
        tokens = _tokenize(expression)
        parser = _Parser(tokens)
        with localcontext() as ctx:
            # Set a reasonable precision and ensure exceptions are raised on invalid ops
            ctx.prec = 28
            ctx.traps[DivisionByZero] = True
            ctx.traps[InvalidOperation] = True
            ctx.traps[Overflow] = True
            ctx.traps[Underflow] = False  # Underflow will subnormalize; not critical here
            ctx.traps[Clamped] = False

            result = parser.parse()

            # Ensure the result is a finite number
            if not result.is_finite():
                return 'Invalid Expression'

        return _format_decimal(result)
    except Exception:
        return 'Invalid Expression'
