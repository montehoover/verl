import re

_ALLOWED_CHARS_RE = re.compile(r'^[\d+\-*/()\s]+$')
_HAS_DIGIT_RE = re.compile(r'\d')

def is_valid_expression(expr: str) -> bool:
    """
    Returns True if expr contains only digits, + - * /, parentheses, and spaces,
    with no disallowed operator sequences or unbalanced parentheses.
    """
    if not isinstance(expr, str):
        return False
    if not expr:
        return False

    # Only allowed characters
    if not _ALLOWED_CHARS_RE.fullmatch(expr):
        return False

    # Disallow exponentiation and integer-division (not in the allowed operator set)
    if '**' in expr or '//' in expr:
        return False

    # Must contain at least one digit
    if not _HAS_DIGIT_RE.search(expr):
        return False

    # Balanced parentheses
    depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                return False
    if depth != 0:
        return False

    return True


def parse_expression(expr: str):
    """
    Parse a mathematical expression string into an AST that respects operator precedence.
    Supports integers, +, -, *, /, parentheses, and unary +/-.
    Returns a nested tuple AST:
      - ('num', value)
      - ('unary', op, node)
      - ('bin', op, left_node, right_node)
    Raises ValueError on invalid input.
    """
    if not is_valid_expression(expr):
        raise ValueError("Expression contains invalid characters or unbalanced parentheses")

    class Parser:
        def __init__(self, s: str):
            self.s = s
            self.i = 0
            self.n = len(s)

        def peek(self):
            self._skip_ws()
            if self.i >= self.n:
                return None
            return self.s[self.i]

        def advance(self):
            ch = self.peek()
            if ch is None:
                return None
            # advance past current non-whitespace character
            while self.i < self.n and self.s[self.i].isspace():
                self.i += 1
            # consume current char
            self.i += 1
            return ch

        def _skip_ws(self):
            while self.i < self.n and self.s[self.i].isspace():
                self.i += 1

        def parse_number(self):
            self._skip_ws()
            start = self.i
            while self.i < self.n and self.s[self.i].isdigit():
                self.i += 1
            if start == self.i:
                raise ValueError(f"Expected number at position {self.i}")
            value = int(self.s[start:self.i])
            return ('num', value)

        def parse_factor(self):
            # Handle unary + or -
            self._skip_ws()
            ch = self.peek()
            if ch in ('+', '-'):
                op = self.advance()
                node = self.parse_factor()
                return ('unary', op, node)

            # Parenthesized expression or number
            ch = self.peek()
            if ch == '(':
                self.advance()  # consume '('
                node = self.parse_expr()
                self._skip_ws()
                if self.peek() != ')':
                    raise ValueError(f"Expected ')' at position {self.i}")
                self.advance()  # consume ')'
                return node

            return self.parse_number()

        def parse_term(self):
            node = self.parse_factor()
            while True:
                self._skip_ws()
                ch = self.peek()
                if ch in ('*', '/'):
                    op = self.advance()
                    right = self.parse_factor()
                    node = ('bin', op, node, right)
                else:
                    break
            return node

        def parse_expr(self):
            node = self.parse_term()
            while True:
                self._skip_ws()
                ch = self.peek()
                if ch in ('+', '-'):
                    op = self.advance()
                    right = self.parse_term()
                    node = ('bin', op, node, right)
                else:
                    break
            return node

    parser = Parser(expr)
    ast = parser.parse_expr()
    # Ensure all input is consumed
    parser._skip_ws()
    if parser.i != parser.n:
        raise ValueError(f"Unexpected character at position {parser.i}")
    return ast
