import re
import math
from typing import List, Tuple, Union

Token = Tuple[str, str]  # (type, value)
UNSAFE_MSG = "Unsafe Operation Detected!"


def _is_safe_expression(expression: str) -> bool:
    if not isinstance(expression, str):
        return False
    if expression is None or len(expression) == 0:
        return False

    # Only allow digits, whitespace, decimal point, basic ops, and parentheses
    allowed_extra = set(".+-*/()")
    for ch in expression:
        if ch.isdigit() or ch.isspace() or ch in allowed_extra:
            continue
        return False

    # Disallow potentially unsafe operator sequences
    if "**" in expression or "//" in expression:
        return False

    # Parentheses must be balanced and never go negative
    balance = 0
    for ch in expression:
        if ch == '(':
            balance += 1
        elif ch == ')':
            balance -= 1
            if balance < 0:
                return False
    if balance != 0:
        return False

    return True


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


def calculate_expression(expression: str):
    """
    Evaluate a simple arithmetic expression string.

    Supports:
    - Addition (+), subtraction (-), multiplication (*), division (/)
    - Operator precedence and parentheses
    - Unary plus/minus (e.g., -3, +4)

    Returns:
    - float result on success
    - 'Unsafe Operation Detected!' on invalid or unsafe input/operation
    """
    if not _is_safe_expression(expression):
        return UNSAFE_MSG

    try:
        tokens = _tokenize(expression)
        parser = _Parser(tokens)
        result = parser.parse()
        if not math.isfinite(result):
            return UNSAFE_MSG
        return result
    except (ValueError, ZeroDivisionError, OverflowError):
        return UNSAFE_MSG
