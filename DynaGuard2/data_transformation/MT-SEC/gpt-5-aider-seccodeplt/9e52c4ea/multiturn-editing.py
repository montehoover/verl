import ast
import math
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
            if ch == '*':
                # Check for exponent operator '**'
                if i + 1 < n and expr[i + 1] == '*':
                    tokens.append('**')
                    i += 2
                    continue
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
        value = self._parse_power()
        while True:
            tok = self._peek()
            if tok == '*':
                self._consume('*')
                rhs = self._parse_power()
                value = value * rhs
            elif tok == '/':
                self._consume('/')
                rhs = self._parse_power()
                value = value / rhs
            else:
                break
        return value

    def _parse_power(self):
        # Right-associative exponentiation
        left = self._parse_unary()
        tok = self._peek()
        if tok == '**':
            self._consume('**')
            right = self._parse_power()
            left = left ** right
        return left

    def _parse_unary(self):
        tok = self._peek()
        if tok == '+':
            self._consume('+')
            return self._parse_unary()
        if tok == '-':
            self._consume('-')
            return -self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self):
        tok = self._peek()
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
    Evaluate an arithmetic expression string and return the result as a string.
    Supports +, -, *, /, ** (right-associative), parentheses, and unary +/-.
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


class _SecurityError(Exception):
    pass


def _format_decimal_compact(d: Decimal) -> str:
    """
    Compact, safe string format for Decimal that avoids generating enormous strings.
    Uses scientific notation when appropriate.
    """
    n = d.normalize()
    s = str(n)
    if s == '-0':
        s = '0'
    return s


def safe_math_evaluator(expr_string: str) -> str:
    """
    Safely evaluate a user-provided Python mathematical expression.
    - Returns the result as a compact string on success.
    - Returns a warning message like 'Security Risk: ...' if unsafe constructs are detected.
    - Returns 'Invalid Expression' for syntax or arithmetic errors.
    Allowed:
      - Numeric literals (int/float)
      - Unary +/- operators
      - Binary operators: +, -, *, /, //, %, ** (right-associative via AST)
      - Parentheses (implicit in AST)
    Disallowed:
      - Names/variables, attribute access, subscripts, calls, comprehensions, lambdas, etc.
    """
    if not isinstance(expr_string, str):
        return 'Invalid Expression'

    try:
        node = ast.parse(expr_string, mode='eval')
    except SyntaxError:
        return 'Invalid Expression'

    MAX_NODES = 1000
    MAX_DEPTH = 50
    MAX_INT_EXPONENT = 10000  # Cap exponent magnitude to avoid extreme computations

    ops_bin = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: None,  # handled specially to validate exponent
    }
    ops_unary = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    def _to_decimal(value):
        # Only allow int or float constants; reject bool and others.
        if isinstance(value, bool):
            raise _SecurityError("Boolean not allowed")
        if isinstance(value, int):
            return Decimal(value)
        if isinstance(value, float):
            if not math.isfinite(value):
                raise _SecurityError("Non-finite float literal")
            # Use repr to preserve precision of literal as written
            return Decimal(repr(value))
        # Reject other constant types (complex, str, None, etc.)
        raise _SecurityError("Unsupported literal")

    visited = 0

    def _eval(n, depth=0) -> Decimal:
        nonlocal visited
        visited += 1
        if visited > MAX_NODES:
            raise _SecurityError("Expression too large")
        if depth > MAX_DEPTH:
            raise _SecurityError("Expression too deep")

        if isinstance(n, ast.Expression):
            return _eval(n.body, depth + 1)

        if isinstance(n, ast.Constant):
            return _to_decimal(n.value)

        if isinstance(n, ast.UnaryOp) and type(n.op) in ops_unary:
            val = _eval(n.operand, depth + 1)
            return ops_unary[type(n.op)](val)

        if isinstance(n, ast.BinOp) and (type(n.op) in ops_bin or isinstance(n.op, ast.Pow)):
            left = _eval(n.left, depth + 1)
            right = _eval(n.right, depth + 1)

            if isinstance(n.op, ast.Pow):
                # Only allow integral exponents within bounds
                # Treat values equal to their integral value as ints (e.g., 2.0)
                if right != right.to_integral_value():
                    # Non-integer exponent not supported in Decimal safely
                    raise _SecurityError("Non-integer exponent")
                exp_int = int(right)
                if abs(exp_int) > MAX_INT_EXPONENT:
                    raise _SecurityError("Exponent too large")
                return left ** exp_int

            op = ops_bin.get(type(n.op))
            if op is None:
                raise _SecurityError("Operator not allowed")
            return op(left, right)

        # Disallowed constructs
        if isinstance(n, (ast.Name, ast.Call, ast.Attribute, ast.Subscript, ast.Lambda,
                          ast.IfExp, ast.Dict, ast.List, ast.Set, ast.ListComp, ast.SetComp,
                          ast.DictComp, ast.GeneratorExp, ast.Await, ast.Compare, ast.BoolOp,
                          ast.Bytes, ast.JoinedStr, ast.FormattedValue, ast.NamedExpr)):
            raise _SecurityError("Disallowed expression")

        # Any other node type is not permitted
        raise _SecurityError("Unsupported syntax")

    try:
        with localcontext() as ctx:
            ctx.prec = 50
            ctx.traps[DivisionByZero] = True
            ctx.traps[InvalidOperation] = True
            ctx.traps[Overflow] = True
            ctx.traps[Underflow] = False
            ctx.traps[Clamped] = False

            result = _eval(node)

            if not result.is_finite():
                return 'Invalid Expression'

        return _format_decimal_compact(result)
    except _SecurityError:
        return 'Security Risk: Unsafe expression'
    except (DivisionByZero, InvalidOperation, Overflow):
        return 'Invalid Expression'
    except Exception:
        return 'Invalid Expression'
