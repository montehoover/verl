from typing import Optional, Union, List, Tuple


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
