import re
import ast
from typing import Optional, Union, List, Tuple


def evaluate_expression(math_expr: str) -> float:
    """
    Safely evaluate a mathematical expression string.

    Supported:
      - Numbers (ints, floats, scientific notation like 1e-3)
      - Operators: +, -, *, /, //, %, ** (with standard precedence)
      - Parentheses: ( )
      - Unary plus/minus

    Returns:
        The evaluated result as a float.

    Raises:
        ValueError: If the expression contains unsupported characters,
                    has invalid syntax, or cannot be safely evaluated.
    """
    if not isinstance(math_expr, str):
        raise ValueError("Expression must be a string.")

    expr = math_expr.strip()
    if not expr:
        raise ValueError("Empty expression.")

    # Allow digits, whitespace, decimal point, operators, parentheses, and scientific notation markers.
    if not re.fullmatch(r"[0-9\s.\+\-\*\/%\(\)eE]+", expr):
        raise ValueError("Expression contains unsupported characters.")

    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError("Invalid expression.") from e

    def _eval(n: ast.AST) -> Union[int, float]:
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        # Numbers
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Unsupported constant.")
        # Compatibility with older Python versions
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            if isinstance(n.n, (int, float)):
                return n.n
            raise ValueError("Unsupported number.")

        # Binary operations
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            op = n.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right
            if isinstance(op, ast.FloorDiv):
                return left // right
            if isinstance(op, ast.Mod):
                return left % right
            if isinstance(op, ast.Pow):
                return left ** right
            raise ValueError("Unsupported operator.")

        # Unary operations
        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator.")

        # Explicitly disallow all other node types (calls, names, attributes, etc.)
        raise ValueError("Unsupported expression.")

    try:
        result = _eval(node)
    except ZeroDivisionError as e:
        raise ValueError("Division by zero.") from e

    return float(result)


def simple_calculator(num1: Union[float, str], num2: Optional[float] = None, operator: Optional[str] = None) -> float:
    """
    Perform basic arithmetic operations.

    Usage modes:
    1) Three-argument mode:
       simple_calculator(4.0, 5.0, '+') -> 9.0
       - num1: First operand (float)
       - num2: Second operand (float)
       - operator: One of '+', '-', '*', '/'

    2) Expression mode (string):
        simple_calculator("4 + 5 * (2 - 1)") -> 9.0
       - Supports +, -, *, / and parentheses.
       - Respects standard operator precedence and parentheses.
       - Raises ValueError on invalid syntax.
       - Raises ZeroDivisionError on division by zero.
    """
    # Expression mode
    if isinstance(num1, str) and num2 is None and operator is None:
        return _evaluate_expression(num1)

    # Three-argument mode (backwards compatible)
    if not isinstance(num1, (int, float)) or not isinstance(num2, (int, float)) or operator is None:
        raise ValueError("Invalid arguments. Provide either (num1: float, num2: float, operator: str) or (expression: str).")

    if operator == '+':
        return float(num1 + num2)
    elif operator == '-':
        return float(num1 - num2)
    elif operator == '*':
        return float(num1 * num2)
    elif operator == '/':
        if float(num2) == 0.0:
            raise ZeroDivisionError("Division by zero.")
        return float(num1 / num2)
    else:
        raise ValueError("Invalid operator. Choose one of '+', '-', '*', '/'.")


def _evaluate_expression(expression: str) -> float:
    tokens = _tokenize(expression)
    value, pos = _parse_expression(tokens, 0)
    if pos != len(tokens):
        raise ValueError("Unexpected input after complete expression.")
    return float(value)


def _tokenize(s: str) -> List[Union[float, str]]:
    tokens: List[Union[float, str]] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if ch in '+-*/()':
            tokens.append(ch)
            i += 1
            continue
        if ch.isdigit() or ch == '.':
            start = i
            has_dot = ch == '.'
            i += 1
            while i < n and (s[i].isdigit() or (s[i] == '.' and not has_dot)):
                if s[i] == '.':
                    has_dot = True
                i += 1
            num_str = s[start:i]
            if num_str in ('.', '+', '-'):
                raise ValueError("Invalid number format.")
            try:
                tokens.append(float(num_str))
            except ValueError:
                raise ValueError(f"Invalid number: {num_str}")
            continue
        # Any other character is invalid
        raise ValueError(f"Invalid character: '{ch}'")
    return tokens


def _parse_expression(tokens: List[Union[float, str]], pos: int) -> Tuple[float, int]:
    # expr := term (('+' | '-') term)*
    value, pos = _parse_term(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ('+', '-'):
        op = tokens[pos]
        rhs, pos = _parse_term(tokens, pos + 1)
        if op == '+':
            value += rhs
        else:
            value -= rhs
    return value, pos


def _parse_term(tokens: List[Union[float, str]], pos: int) -> Tuple[float, int]:
    # term := factor (('*' | '/') factor)*
    value, pos = _parse_factor(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ('*', '/'):
        op = tokens[pos]
        rhs, pos = _parse_factor(tokens, pos + 1)
        if op == '*':
            value *= rhs
        else:
            if rhs == 0.0:
                raise ZeroDivisionError("Division by zero.")
            value /= rhs
    return value, pos


def _parse_factor(tokens: List[Union[float, str]], pos: int) -> Tuple[float, int]:
    # factor := NUMBER | '(' expr ')' | ('+' | '-') factor   (unary plus/minus)
    if pos >= len(tokens):
        raise ValueError("Unexpected end of expression.")

    tok = tokens[pos]

    # Unary operators
    if tok == '+':
        return _parse_factor(tokens, pos + 1)
    if tok == '-':
        val, new_pos = _parse_factor(tokens, pos + 1)
        return -val, new_pos

    # Parenthesized expression
    if tok == '(':
        value, new_pos = _parse_expression(tokens, pos + 1)
        if new_pos >= len(tokens) or tokens[new_pos] != ')':
            raise ValueError("Missing closing parenthesis.")
        return value, new_pos + 1

    # Number
    if isinstance(tok, (int, float)):
        return float(tok), pos + 1

    raise ValueError("Expected a number, unary operator, or '('.")
