import re
import ast
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


_ALLOWED_CHARS_REGEX = re.compile(r'^[\d\s+\-*/().eE]+$')

def parse_and_calculate(expression: str) -> float:
    """
    Parse and safely evaluate a mathematical expression.

    Supports:
      - Numbers (ints, floats, scientific notation)
      - Binary operators: +, -, *, /
      - Unary operators: +, -
      - Parentheses: ()

    Args:
        expression: The expression string to evaluate.

    Returns:
        The result of the calculation as a float.

    Raises:
        ValueError: If the expression contains unsupported characters, uses
                    unsupported operations, is syntactically invalid, or
                    cannot be computed safely (e.g., division by zero).
    """
    if not isinstance(expression, str) or expression.strip() == "":
        raise ValueError("Expression must be a non-empty string.")

    if not _ALLOWED_CHARS_REGEX.match(expression):
        raise ValueError("Unsupported characters in expression.")

    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError:
        raise ValueError("Invalid expression syntax.")

    def _eval(node) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric constants are allowed.")

        # For Python versions where numbers are represented by ast.Num
        if hasattr(ast, 'Num') and isinstance(node, getattr(ast, 'Num')):
            return float(node.n)

        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)

            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                if right == 0.0:
                    raise ValueError("Division by zero is not allowed.")
                result = left / right
            else:
                raise ValueError("Unsupported binary operator.")
            return float(result)

        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return float(+operand)
            if isinstance(node.op, ast.USub):
                return float(-operand)
            raise ValueError("Unsupported unary operator.")

        # Any other AST node type is disallowed
        raise ValueError("Unsupported expression construct.")

    result = _eval(tree)

    # Reject non-finite results (inf, -inf, NaN)
    if result == float('inf') or result == float('-inf') or result != result:
        raise ValueError("Computation resulted in a non-finite value.")

    return float(result)
