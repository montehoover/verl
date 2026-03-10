import re
from typing import List, Dict, Union, Optional

def sanitize_input(expression: str) -> bool:
    """
    Validate that the input string contains only digits, arithmetic operators
    (+, -, *, /), parentheses, and whitespace; and that it forms a basic, safe
    arithmetic expression.

    Returns:
        bool: True if the input is safe and structurally valid, False otherwise.

    Raises:
        ValueError: If any invalid characters are found.
    """
    if not isinstance(expression, str):
        raise ValueError("Input must be a string.")

    # If there are any characters outside the allowed set, raise ValueError.
    invalid_chars = re.findall(r"[^\d+\-*/()\s]", expression)
    if invalid_chars:
        unique = sorted(set(invalid_chars))
        raise ValueError(f"Invalid character(s) found: {''.join(unique)}")

    # Strip spaces for structural checks.
    expr = re.sub(r"\s+", "", expression)

    # Empty or whitespace-only is not a valid expression.
    if not expr:
        return False

    # Check balanced parentheses.
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

    # Simple structural validation:
    # - Numbers are sequences of digits.
    # - Operators are + - * /
    # - Allow unary + or - at the start, or immediately after '(' or another operator.
    last = 'start'  # one of: start, num, op, lp, rp, op_unary
    i = 0
    n = len(expr)

    while i < n:
        ch = expr[i]

        if ch.isdigit():
            # Consume the full number (integers only, no decimals).
            while i + 1 < n and expr[i + 1].isdigit():
                i += 1
            last = 'num'

        elif ch == '(':
            if last not in ('start', 'op', 'lp', 'op_unary'):
                return False
            last = 'lp'

        elif ch == ')':
            if last not in ('num', 'rp'):
                return False
            last = 'rp'

        elif ch in '+-*/':
            if last in ('num', 'rp'):
                # Binary operator is valid after a number or right paren.
                last = 'op'
            else:
                # Allow unary + or - at start, after (, or after another operator.
                if ch in '+-' and last in ('start', 'lp', 'op', 'op_unary'):
                    last = 'op_unary'
                else:
                    return False
        else:
            # Should be unreachable due to the earlier character filter.
            return False

        i += 1

    # Expression should end with a number or a right parenthesis.
    return last in ('num', 'rp')


Token = Dict[str, Union[str, int]]

def parse_expression(expression: str) -> List[Token]:
    """
    Parse a validated arithmetic expression string into Reverse Polish Notation (RPN)
    tokens preserving the correct order of operations and parentheses.

    Tokens are dictionaries with:
    - {'type': 'number', 'value': int}
    - {'type': 'operator', 'value': one of '+', '-', '*', '/', 'NEG', 'arity': 1 or 2}

    'NEG' represents unary negation. Unary plus is treated as a no-op and omitted.

    Raises:
        ValueError: If the expression is not a valid arithmetic expression per sanitize_input.
    """
    if not isinstance(expression, str):
        raise ValueError("Input must be a string.")

    # Ensure the input is valid before parsing.
    if not sanitize_input(expression):
        raise ValueError("Invalid arithmetic expression.")

    # Remove whitespace for tokenization.
    expr = re.sub(r"\s+", "", expression)

    # Tokenize into numbers, operators, and parentheses, detecting unary operators.
    raw_tokens: List[Token] = []
    i = 0
    n = len(expr)
    prev_type: Optional[str] = None  # None, 'number', 'operator', 'lparen', 'rparen'

    def at_unary_position(prev_t: Optional[str]) -> bool:
        return prev_t in (None, 'operator', 'lparen')

    while i < n:
        ch = expr[i]

        if ch.isdigit():
            start = i
            while i + 1 < n and expr[i + 1].isdigit():
                i += 1
            value = int(expr[start:i + 1])
            raw_tokens.append({'type': 'number', 'value': value})
            prev_type = 'number'

        elif ch in '+-*/':
            if ch in '+-' and at_unary_position(prev_type):
                # Unary operator
                if ch == '-':
                    raw_tokens.append({'type': 'operator', 'value': 'NEG', 'arity': 1})
                # Unary plus is a no-op; skip emitting a token.
                prev_type = 'operator'
            else:
                # Binary operator
                raw_tokens.append({'type': 'operator', 'value': ch, 'arity': 2})
                prev_type = 'operator'

        elif ch == '(':
            raw_tokens.append({'type': 'lparen', 'value': '('})
            prev_type = 'lparen'

        elif ch == ')':
            raw_tokens.append({'type': 'rparen', 'value': ')'})
            prev_type = 'rparen'

        else:
            # Should not occur due to prior validation
            raise ValueError(f"Unexpected character '{ch}' during parsing.")

        i += 1

    # Shunting Yard to convert to RPN
    output: List[Token] = []
    op_stack: List[Token] = []

    precedence = {
        '+': 1,
        '-': 1,
        '*': 2,
        '/': 2,
        'NEG': 3,  # unary negation has higher precedence
    }
    associativity = {
        '+': 'left',
        '-': 'left',
        '*': 'left',
        '/': 'left',
        'NEG': 'right',  # unary operators are typically right-associative
    }

    for tok in raw_tokens:
        t = tok['type']

        if t == 'number':
            output.append(tok)

        elif t == 'operator':
            p = precedence[tok['value']]
            assoc = associativity[tok['value']]

            while op_stack:
                top = op_stack[-1]
                if top['type'] != 'operator':
                    break
                top_p = precedence[top['value']]
                if (assoc == 'left' and p <= top_p) or (assoc == 'right' and p < top_p):
                    output.append(op_stack.pop())
                else:
                    break
            op_stack.append(tok)

        elif t == 'lparen':
            op_stack.append(tok)

        elif t == 'rparen':
            # Pop operators until matching '('
            while op_stack and op_stack[-1]['type'] != 'lparen':
                output.append(op_stack.pop())
            if not op_stack:
                raise ValueError("Mismatched parentheses.")
            # Discard the '('
            op_stack.pop()

        else:
            raise ValueError(f"Unknown token type: {t}")

    # Drain the operator stack
    while op_stack:
        top = op_stack.pop()
        if top['type'] in ('lparen', 'rparen'):
            raise ValueError("Mismatched parentheses.")
        output.append(top)

    return output


def safe_eval_expression(expression: str) -> Union[int, float]:
    """
    Safely evaluate an arithmetic expression string.

    Steps:
    - Validate with sanitize_input (raises ValueError on invalid/unsafe input).
    - Parse into RPN tokens with parse_expression.
    - Evaluate the RPN expression using an explicit stack.

    Returns:
        int or float: The evaluated result.

    Raises:
        ValueError: If the input is unsafe, malformed, or evaluation fails.
    """
    if not isinstance(expression, str):
        raise ValueError("Input must be a string.")

    # Validate input; sanitize_input returns bool or raises ValueError for bad chars.
    try:
        is_safe = sanitize_input(expression)
    except ValueError as e:
        raise ValueError(str(e))
    if not is_safe:
        raise ValueError("Invalid or empty arithmetic expression.")

    # Parse to RPN tokens (parse_expression also re-validates).
    tokens = parse_expression(expression)

    stack: List[Union[int, float]] = []

    for tok in tokens:
        ttype = tok.get('type')
        if ttype == 'number':
            stack.append(tok['value'])  # int
        elif ttype == 'operator':
            op = tok['value']
            arity = tok.get('arity', 2)
            try:
                if arity == 1:
                    if len(stack) < 1:
                        raise ValueError("Malformed expression: missing operand for unary operator.")
                    a = stack.pop()
                    if op == 'NEG':
                        stack.append(-a)
                    else:
                        raise ValueError(f"Unknown unary operator: {op}")
                elif arity == 2:
                    if len(stack) < 2:
                        raise ValueError("Malformed expression: missing operand for binary operator.")
                    b = stack.pop()
                    a = stack.pop()
                    if op == '+':
                        stack.append(a + b)
                    elif op == '-':
                        stack.append(a - b)
                    elif op == '*':
                        stack.append(a * b)
                    elif op == '/':
                        if b == 0:
                            raise ValueError("Division by zero.")
                        stack.append(a / b)
                    else:
                        raise ValueError(f"Unknown operator: {op}")
                else:
                    raise ValueError(f"Unsupported operator arity: {arity}")
            except ZeroDivisionError:
                raise ValueError("Division by zero.")
        else:
            raise ValueError(f"Unexpected token type during evaluation: {ttype}")

    if len(stack) != 1:
        raise ValueError("Malformed expression: leftover operands/operators.")

    return stack[0]
