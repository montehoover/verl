from typing import Optional

def is_valid_expression(expr: str) -> bool:
    """
    Validate whether the input string is a safe arithmetic expression consisting only of:
      - numbers (integers or decimals like 123 or 123.456)
      - operators: +, -, *, /
      - parentheses: (, )
      - whitespace

    Constraints and behavior:
      - No evaluation is performed.
      - Unary + and - are allowed (e.g., -3, +4, 1 + -2).
      - Decimal numbers must have at least one digit before and after the decimal point (e.g., 0.5 is valid, .5 and 5. are invalid).
      - No other characters (including letters, underscores, exponent notation) are allowed.
      - Balanced parentheses are required.
      - The expression must be syntactically valid (no trailing operators, empty parentheses, etc.).

    Returns:
        True if the expression is valid and safe; False otherwise.
    """
    if not isinstance(expr, str):
        return False

    # Basic resource limits to avoid pathological inputs
    MAX_LENGTH = 10000
    MAX_NESTING = 1000

    if len(expr) == 0 or len(expr) > MAX_LENGTH:
        return False

    # Quick character whitelist check
    allowed_chars = set("0123456789+-*/(). \t\n\r\f\v")
    if any(ch not in allowed_chars for ch in expr):
        return False

    # Strip whitespace to check for completely empty input after spaces
    if expr.strip() == "":
        return False

    class Parser:
        def __init__(self, s: str):
            self.s = s
            self.i = 0
            self.n = len(s)
            self.paren_depth = 0
            self.error = False

        def peek(self) -> Optional[str]:
            # Skip whitespace
            while self.i < self.n and self.s[self.i].isspace():
                self.i += 1
            if self.i >= self.n:
                return None
            return self.s[self.i]

        def consume(self, ch: str) -> bool:
            if self.peek() == ch:
                self.i += 1
                return True
            return False

        def parse_number(self) -> bool:
            # number := DIGITS ('.' DIGITS)?
            start = self.i
            # Consume leading whitespace in peek
            ch = self.peek()
            if ch is None or not ch.isdigit():
                return False
            # digits
            while self.i < self.n and self.s[self.i].isdigit():
                self.i += 1
            # optional decimal part
            if self.i < self.n and self.s[self.i] == '.':
                # ensure at least one digit before and after '.'
                # We already consumed at least one digit before.
                self.i += 1
                if self.i >= self.n or not self.s[self.i].isdigit():
                    # no digits after the dot
                    self.i = start
                    return False
                while self.i < self.n and self.s[self.i].isdigit():
                    self.i += 1
            # ensure no accidental second dot adjacent (e.g., "1..2")
            return True

        def parse_factor(self) -> bool:
            # factor := NUMBER | '(' expr ')' | ('+'|'-') factor
            ch = self.peek()
            if ch is None:
                return False

            # Unary + or -
            if ch in "+-":
                # consume the sign and parse the next factor
                self.i += 1
                return self.parse_factor()

            # Parenthesized expression
            if ch == '(':
                self.i += 1
                self.paren_depth += 1
                if self.paren_depth > MAX_NESTING:
                    return False
                if not self.parse_expr():
                    return False
                # Must close with ')'
                if not self.consume(')'):
                    return False
                self.paren_depth -= 1
                return True

            # Number
            if ch.isdigit():
                return self.parse_number()

            # Anything else is invalid
            return False

        def parse_term(self) -> bool:
            # term := factor (('*'|'/') factor)*
            if not self.parse_factor():
                return False
            while True:
                ch = self.peek()
                if ch in ('*', '/'):
                    # consume operator
                    self.i += 1
                    if not self.parse_factor():
                        return False
                else:
                    break
            return True

        def parse_expr(self) -> bool:
            # expr := term (('+'|'-') term)*
            if not self.parse_term():
                return False
            while True:
                ch = self.peek()
                if ch in ('+', '-'):
                    # consume operator
                    self.i += 1
                    if not self.parse_term():
                        return False
                else:
                    break
            return True

    parser = Parser(expr)
    ok = parser.parse_expr()
    # Ensure full consumption
    if not ok:
        return False
    # After parsing, ensure no trailing non-whitespace characters remain
    if parser.peek() is not None:
        return False
    # Balanced parentheses check
    if parser.paren_depth != 0:
        return False
    return True
