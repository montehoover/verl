import re
from typing import Optional

class _ExpressionParser:
    TOKEN_REGEX = re.compile(
        r'\s*(?:(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|(?P<op>[+\-*/()]))'
    )

    def __init__(self, text: str):
        self.tokens = self._tokenize(text)
        self.pos = 0

    def _tokenize(self, text: str):
        tokens = []
        pos = 0
        n = len(text)
        while pos < n:
            m = self.TOKEN_REGEX.match(text, pos)
            if not m:
                # If it's whitespace, skip it; otherwise invalid character
                if text[pos].isspace():
                    pos += 1
                    continue
                raise ValueError(f"Invalid character at position {pos}: '{text[pos]}'")
            pos = m.end()
            if m.group('number') is not None:
                tokens.append(('NUMBER', float(m.group('number'))))
            else:
                tokens.append(m.group('op'))
        return tokens

    def _peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self, expected=None):
        tok = self._peek()
        if tok is None:
            return None
        if expected is not None:
            if tok == expected or (isinstance(tok, tuple) and tok[0] == expected):
                self.pos += 1
                return tok
            raise ValueError(f"Expected {expected}, got {tok}")
        self.pos += 1
        return tok

    def parse(self) -> float:
        result = self._parse_expression()
        if self._peek() is not None:
            raise ValueError("Unexpected input after end of expression.")
        return result

    def _parse_expression(self) -> float:
        value = self._parse_term()
        while True:
            tok = self._peek()
            if tok == '+' or tok == '-':
                op = self._consume()
                rhs = self._parse_term()
                if op == '+':
                    value += rhs
                else:
                    value -= rhs
            else:
                break
        return value

    def _parse_term(self) -> float:
        value = self._parse_factor()
        while True:
            tok = self._peek()
            if tok == '*' or tok == '/':
                op = self._consume()
                rhs = self._parse_factor()
                if op == '*':
                    value *= rhs
                else:
                    if rhs == 0:
                        raise ZeroDivisionError("Division by zero is not allowed.")
                    value /= rhs
            else:
                break
        return value

    def _parse_factor(self) -> float:
        tok = self._peek()
        # Unary plus/minus
        if tok == '+' or tok == '-':
            op = self._consume()
            val = self._parse_factor()
            return val if op == '+' else -val

        if tok == '(':
            self._consume('(')
            val = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Missing closing parenthesis.")
            self._consume(')')
            return val

        if isinstance(tok, tuple) and tok[0] == 'NUMBER':
            number = self._consume('NUMBER')[1]
            return number

        raise ValueError(f"Unexpected token: {tok}")

def _evaluate_expression(expression: str) -> float:
    parser = _ExpressionParser(expression)
    return float(parser.parse())

def basic_calculator(num1: Optional[float] = None,
                     num2: Optional[float] = None,
                     operator: Optional[str] = None,
                     expression: Optional[str] = None) -> float:
    """
    Perform a basic arithmetic operation.

    Usage:
      - As a simple calculator with explicit operands and operator:
            basic_calculator(4, 3, '+') -> 7.0
      - As an expression evaluator with parentheses and precedence:
            basic_calculator(expression="(4 + 3) * 2") -> 14.0

    Args:
        num1: The first operand (used with operator).
        num2: The second operand (used with operator).
        operator: One of '+', '-', '*', '/' (used with num1 and num2).
        expression: A string expression containing numbers, '+', '-', '*', '/',
                    and parentheses '()'.

    Returns:
        The result of the calculation as a float.

    Raises:
        ValueError: If inputs are invalid or operator is not supported.
        ZeroDivisionError: If division by zero is attempted.
    """
    if expression is not None:
        if not isinstance(expression, str) or expression.strip() == "":
            raise ValueError("Expression must be a non-empty string.")
        return _evaluate_expression(expression)

    if operator in ('+', '-', '*', '/'):
        if num1 is None or num2 is None:
            raise ValueError("num1 and num2 must be provided when using operator.")
        if operator == '+':
            return float(num1 + num2)
        elif operator == '-':
            return float(num1 - num2)
        elif operator == '*':
            return float(num1 * num2)
        elif operator == '/':
            if num2 == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            return float(num1 / num2)

    raise ValueError("Provide either num1, num2, and a valid operator ('+', '-', '*', '/'), or a valid expression string.")
