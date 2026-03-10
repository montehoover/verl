import re
from typing import List, Tuple, Union

Token = Tuple[str, str]  # (type, value)


def _tokenize(expression: str) -> List[Token]:
    if expression is None:
        raise ValueError("Expression cannot be None")

    token_specification = [
        ("NUMBER", r'(?:\d+(?:\.\d*)?|\.\d+)'),
        ("OP", r'[+\-*/]'),
        ("LPAREN", r'\('),
        ("RPAREN", r'\)'),
        ("SKIP", r'\s+'),
        ("MISMATCH", r'.'),
    ]
    tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)
    tokens: List[Token] = []
    for mo in re.finditer(tok_regex, expression):
        kind = mo.lastgroup
        value = mo.group()
        if kind == "SKIP":
            continue
        elif kind == "MISMATCH":
            raise ValueError(f"Unexpected character '{value}' at position {mo.start()}")
        else:
            tokens.append((kind, value))
    tokens.append(("EOF", ""))  # End marker
    return tokens


class _Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> None:
        if self.pos < len(self.tokens) - 1:
            self.pos += 1

    def expect(self, kind: str) -> Token:
        tok = self.current()
        if tok[0] != kind:
            raise ValueError(f"Expected {kind}, got {tok[0]}")
        self.advance()
        return tok

    def parse(self) -> float:
        if self.current()[0] == "EOF":
            raise ValueError("Empty expression")
        value = self.parse_expression()
        if self.current()[0] != "EOF":
            raise ValueError("Unexpected input after complete expression")
        return value

    # expression -> term ((+|-) term)*
    def parse_expression(self) -> float:
        value = self.parse_term()
        while self.current()[0] == "OP" and self.current()[1] in ('+', '-'):
            op = self.current()[1]
            self.advance()
            rhs = self.parse_term()
            if op == '+':
                value += rhs
            else:
                value -= rhs
        return value

    # term -> factor ((*|/) factor)*
    def parse_term(self) -> float:
        value = self.parse_factor()
        while self.current()[0] == "OP" and self.current()[1] in ('*', '/'):
            op = self.current()[1]
            self.advance()
            rhs = self.parse_factor()
            if op == '*':
                value *= rhs
            else:
                if rhs == 0:
                    raise ZeroDivisionError("division by zero")
                value /= rhs
        return value

    # factor -> NUMBER | '(' expression ')' | (+|-) factor
    def parse_factor(self) -> float:
        tok_type, tok_val = self.current()

        # Unary plus/minus
        if tok_type == "OP" and tok_val in ('+', '-'):
            op = tok_val
            self.advance()
            val = self.parse_factor()
            return val if op == '+' else -val

        if tok_type == "NUMBER":
            self.advance()
            return float(tok_val)

        if tok_type == "LPAREN":
            self.advance()
            val = self.parse_expression()
            self.expect("RPAREN")
            return val

        raise ValueError(f"Unexpected token: {tok_type} '{tok_val}'")


def calculate_expression(expression: str) -> float:
    """
    Evaluate a simple arithmetic expression string.

    Supports:
    - Addition (+), subtraction (-), multiplication (*), division (/)
    - Operator precedence and parentheses
    - Unary plus/minus (e.g., -3, +4)

    Returns a float result. Raises ValueError on invalid syntax
    and ZeroDivisionError on division by zero.
    """
    tokens = _tokenize(expression)
    parser = _Parser(tokens)
    return parser.parse()
