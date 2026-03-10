import re
from typing import Optional, Tuple, Union

def simple_calculate(num1: Union[int, float, str], num2: Optional[Union[int, float]] = None, operator: Optional[str] = None) -> float:
    """
    Perform a basic arithmetic operation on two numbers.

    Usage:
        - simple_calculate(num1, num2, operator)
        - simple_calculate("7 + 8")

    Args:
        num1: First number, or an expression string like "7 + 8".
        num2: Second number (ignored if num1 is a string expression).
        operator: One of '+', '-', '*', '/' (ignored if num1 is a string expression).

    Returns:
        The result as a float.

    Raises:
        ValueError: If the operator or expression is not supported/valid.
        ZeroDivisionError: If division by zero is attempted.
    """

    def _parse_expression(expr: str) -> Tuple[float, float, str]:
        pattern = r'^\s*([+-]?\d+(?:\.\d+)?)\s*([+\-*/])\s*([+-]?\d+(?:\.\d+)?)\s*$'
        match = re.match(pattern, expr)
        if not match:
            raise ValueError('Invalid expression format. Expected like "7 + 8".')
        left_str, op, right_str = match.groups()
        return float(left_str), float(right_str), op

    if isinstance(num1, str) and num2 is None and operator is None:
        a, b, op = _parse_expression(num1)
    else:
        a = float(num1)
        if num2 is None or operator is None:
            raise ValueError("When providing numeric arguments, both num2 and operator are required.")
        b = float(num2)
        op = operator

    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        if b == 0.0:
            raise ZeroDivisionError("Division by zero.")
        return a / b
    else:
        raise ValueError(f"Unsupported operator: {op}. Use one of '+', '-', '*', '/'.")
        

def process_user_query(query: str) -> float:
    """
    Safely evaluate a mathematical expression from a user-provided string.

    Supports:
      - Numbers (integers, decimals, and scientific notation like 1e3)
      - Operators: +, -, *, /
      - Parentheses: ( )

    Args:
        query: The user-provided mathematical expression.

    Returns:
        The computed result as a float.

    Raises:
        ValueError: If the input contains unsafe characters, is malformed,
                    has mismatched parentheses, or attempts division by zero.
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")

    s = query.strip()
    if not s:
        raise ValueError("Empty query.")

    # Allow only digits, whitespace, operators, parentheses, decimal point, and exponent markers (e/E).
    if not re.match(r'^[\d\s+\-*/().eE]+$', s):
        raise ValueError("Query contains unsafe characters.")

    # Tokenization with support for unary +/-
    number_re = re.compile(r'^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
    tokens = []
    i = 0
    n = len(s)
    prev_type = 'start'  # one of: start, number, op, '(', ')'

    def _append_operator(op_char: str):
        nonlocal prev_type
        if prev_type in ('start', 'op', '('):
            raise ValueError("Malformed expression: operator in invalid position.")
        tokens.append(op_char)
        prev_type = 'op'

    # Track parentheses balance during tokenization
    paren_balance = 0

    while i < n:
        ch = s[i]

        if ch.isspace():
            i += 1
            continue

        # Number (unsigned)
        if ch.isdigit() or ch == '.':
            m = number_re.match(s[i:])
            if not m:
                raise ValueError("Malformed number.")
            tokens.append(float(m.group(0)))
            i += len(m.group(0))
            prev_type = 'number'
            continue

        # Parentheses
        if ch == '(':
            tokens.append('(')
            paren_balance += 1
            i += 1
            prev_type = '('
            continue

        if ch == ')':
            if paren_balance == 0:
                raise ValueError("Mismatched parentheses.")
            # Disallow empty parentheses like "()" or "(+)"
            if prev_type in ('op', '(', 'start'):
                raise ValueError("Malformed expression: empty or invalid parentheses.")
            tokens.append(')')
            paren_balance -= 1
            i += 1
            prev_type = ')'
            continue

        # Operators, including unary +/-
        if ch in '+-':
            if prev_type in ('start', 'op', '('):
                # Unary +/-
                sign = -1.0 if ch == '-' else 1.0
                i += 1
                # Skip whitespace after unary sign
                while i < n and s[i].isspace():
                    i += 1
                if i >= n:
                    raise ValueError("Malformed expression: dangling unary operator.")

                # Unary before parenthesis: -( ... ) -> 0 - ( ... ), +( ... ) -> 0 + ( ... )
                if s[i] == '(':
                    tokens.append(0.0)
                    tokens.append('-' if sign < 0 else '+')
                    tokens.append('(')
                    paren_balance += 1
                    i += 1
                    prev_type = '('
                    continue

                # Expect a number after unary sign
                m = number_re.match(s[i:])
                if not m:
                    raise ValueError("Malformed expression: expected number after unary operator.")
                val = float(m.group(0)) * sign
                tokens.append(val)
                i += len(m.group(0))
                prev_type = 'number'
                continue
            else:
                # Binary +/-
                _append_operator(ch)
                i += 1
                continue

        if ch in '*/':
            if prev_type in ('start', 'op', '('):
                raise ValueError("Malformed expression: operator in invalid position.")
            tokens.append(ch)
            i += 1
            prev_type = 'op'
            continue

        # Any other character should have been blocked by the allowed-charset check
        raise ValueError("Query contains unsafe or invalid characters.")

    if paren_balance != 0:
        raise ValueError("Mismatched parentheses.")

    # Convert to Reverse Polish Notation (Shunting-yard)
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output = []
    op_stack = []

    for tok in tokens:
        if isinstance(tok, float):
            output.append(tok)
        elif tok in precedence:
            while op_stack and op_stack[-1] in precedence and precedence[op_stack[-1]] >= precedence[tok]:
                output.append(op_stack.pop())
            op_stack.append(tok)
        elif tok == '(':
            op_stack.append(tok)
        elif tok == ')':
            while op_stack and op_stack[-1] != '(':
                output.append(op_stack.pop())
            if not op_stack:
                raise ValueError("Mismatched parentheses.")
            op_stack.pop()  # discard '('
        else:
            raise ValueError("Malformed expression.")

    while op_stack:
        top = op_stack.pop()
        if top == '(':
            raise ValueError("Mismatched parentheses.")
        output.append(top)

    # Evaluate RPN
    stack = []
    try:
        for tok in output:
            if isinstance(tok, float):
                stack.append(tok)
            else:
                if len(stack) < 2:
                    raise ValueError("Malformed expression.")
                b = stack.pop()
                a = stack.pop()
                if tok == '+':
                    stack.append(a + b)
                elif tok == '-':
                    stack.append(a - b)
                elif tok == '*':
                    stack.append(a * b)
                elif tok == '/':
                    if b == 0.0:
                        raise ValueError("Division by zero.")
                    stack.append(a / b)
                else:
                    raise ValueError("Unsupported operator.")
        if len(stack) != 1:
            raise ValueError("Malformed expression.")
        return float(stack[0])
    except ZeroDivisionError:
        # Normalize to ValueError per requirements
        raise ValueError("Division by zero.")
