import re
import ast
from typing import List, Union

# Regex allows only digits, spaces, +, -, *, /, and parentheses.
# It also disallows "**" (exponent) and "//" (floor division) which are not basic operators.
_SAFE_EXPR_RE = re.compile(r'^(?!.*\*\*)(?!.*//)[\d\s+\-*/()]+$')

def sanitize_expression(expr: str) -> bool:
    """
    Validate that the expression contains only numbers, basic arithmetic operators (+, -, *, /),
    parentheses, and spaces. Disallows '**' and '//' to avoid non-basic operations.

    Returns True if valid, False otherwise.
    """
    if not isinstance(expr, str):
        return False

    if not expr:
        return False

    if _SAFE_EXPR_RE.fullmatch(expr) is None:
        return False

    # Optional additional check: ensure parentheses are balanced
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


Token = Union[int, str]  # int for numbers, str for operators or parentheses during parsing

def parse_expression(expr: str) -> List[Token]:
    """
    Parse a valid arithmetic expression into a structured representation that respects operator precedence.
    Returns tokens in Reverse Polish Notation (RPN), where integers represent numbers and strings represent
    operators ('+', '-', '*', '/', 'u-', 'u+' for unary minus/plus).

    This function does not evaluate the expression. It only parses structure.

    Raises:
        ValueError: If the expression is syntactically invalid (e.g., mismatched operators/operands).
    """
    tokens = _tokenize(expr)
    rpn = _to_rpn(tokens)
    _validate_rpn(rpn)
    return rpn


def _tokenize(expr: str) -> List[Token]:
    """
    Convert the input string into a list of tokens (integers and operator/parenthesis strings).
    Supports unary plus/minus by emitting 'u+' and 'u-' tokens.
    """
    tokens: List[Token] = []
    i = 0
    n = len(expr)
    prev_type = 'start'  # one of: start, number, rparen, operator, lparen

    while i < n:
        ch = expr[i]

        if ch.isspace():
            i += 1
            continue

        if ch.isdigit():
            start = i
            while i < n and expr[i].isdigit():
                i += 1
            num = int(expr[start:i])
            tokens.append(num)
            prev_type = 'number'
            continue

        if ch in '+-':
            # Determine unary vs binary
            is_unary = prev_type in ('start', 'operator', 'lparen')
            if ch == '+' and is_unary:
                tokens.append('u+')
                prev_type = 'operator'
                i += 1
                continue
            if ch == '-' and is_unary:
                tokens.append('u-')
                prev_type = 'operator'
                i += 1
                continue
            # Binary + or -
            tokens.append(ch)
            prev_type = 'operator'
            i += 1
            continue

        if ch in '*/':
            tokens.append(ch)
            prev_type = 'operator'
            i += 1
            continue

        if ch == '(':
            tokens.append(ch)
            prev_type = 'lparen'
            i += 1
            continue

        if ch == ')':
            tokens.append(ch)
            prev_type = 'rparen'
            i += 1
            continue

        # Any other character should not appear if sanitized; treat as error defensively.
        raise ValueError(f"Invalid character in expression: {ch!r}")

    return tokens


def _to_rpn(tokens: List[Token]) -> List[Token]:
    """
    Convert token list to Reverse Polish Notation using the shunting-yard algorithm.
    Supports unary plus/minus with higher precedence and right-associativity.
    """
    output: List[Token] = []
    ops: List[str] = []

    precedence = {
        'u+': 3,
        'u-': 3,
        '*': 2,
        '/': 2,
        '+': 1,
        '-': 1,
    }
    right_assoc = {'u+', 'u-'}  # unary operators are right-associative

    def is_op(tok: Token) -> bool:
        return isinstance(tok, str) and tok in precedence

    for tok in tokens:
        if isinstance(tok, int):
            output.append(tok)
        elif tok == '(':
            ops.append(tok)
        elif tok == ')':
            while ops and ops[-1] != '(':
                output.append(ops.pop())
            if not ops or ops[-1] != '(':
                raise ValueError("Mismatched parentheses")
            ops.pop()  # discard '('
        elif is_op(tok):
            # Pop operators from stack respecting precedence and associativity
            while ops and is_op(ops[-1]):
                top = ops[-1]
                if (top not in right_assoc and precedence[top] >= precedence[tok]) or \
                   (top in right_assoc and precedence[top] > precedence[tok]):
                    output.append(ops.pop())
                else:
                    break
            ops.append(tok)
        else:
            raise ValueError(f"Unexpected token: {tok!r}")

    while ops:
        op = ops.pop()
        if op in ('(', ')'):
            raise ValueError("Mismatched parentheses")
        output.append(op)

    return output


def _validate_rpn(rpn: List[Token]) -> None:
    """
    Validate that the RPN sequence is syntactically correct in terms of operand/operator counts.
    Ensures that evaluation would consume operands correctly and leave exactly one result.
    """
    stack_depth = 0
    for tok in rpn:
        if isinstance(tok, int):
            stack_depth += 1
        elif tok in ('u+', 'u-'):
            # unary operator: needs 1 operand
            if stack_depth < 1:
                raise ValueError("Unary operator lacks operand")
            # consumes 1, produces 1 -> depth unchanged
        elif tok in ('+', '-', '*', '/'):
            # binary operator: needs 2 operands
            if stack_depth < 2:
                raise ValueError("Binary operator lacks operands")
            stack_depth -= 1  # consumes two, produces one
        else:
            raise ValueError(f"Invalid token in RPN: {tok!r}")

    if stack_depth != 1:
        raise ValueError("Expression is incomplete or malformed")


def evaluate_and_compute(expr: str) -> Union[int, float]:
    """
    Validate, parse, and evaluate the arithmetic expression string.
    Returns the computed numeric result.
    Raises ValueError for invalid input.
    """
    if not sanitize_expression(expr):
        raise ValueError("Invalid expression")

    try:
        rpn = parse_expression(expr)
    except ValueError as e:
        raise ValueError("Invalid expression") from e

    stack: List[Union[int, float]] = []
    for tok in rpn:
        if isinstance(tok, int):
            stack.append(tok)
        elif tok == 'u+':
            if not stack:
                raise ValueError("Invalid expression")
            a = stack.pop()
            stack.append(+a)
        elif tok == 'u-':
            if not stack:
                raise ValueError("Invalid expression")
            a = stack.pop()
            stack.append(-a)
        elif tok in ('+', '-', '*', '/'):
            if len(stack) < 2:
                raise ValueError("Invalid expression")
            b = stack.pop()
            a = stack.pop()
            if tok == '+':
                stack.append(a + b)
            elif tok == '-':
                stack.append(a - b)
            elif tok == '*':
                stack.append(a * b)
            elif tok == '/':
                stack.append(a / b)
        else:
            raise ValueError("Invalid expression")

    if len(stack) != 1:
        raise ValueError("Invalid expression")

    return stack[0]
