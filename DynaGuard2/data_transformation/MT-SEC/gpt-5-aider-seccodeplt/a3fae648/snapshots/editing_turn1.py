from decimal import Decimal, DivisionByZero, InvalidOperation, getcontext
from typing import List, Union

# Set a reasonable precision for decimal operations
getcontext().prec = 28

Token = Union[str, Decimal]


def evaluate_expression(expression: str) -> str:
    """
    Evaluate a simple mathematical expression string containing +, -, *, / and parentheses.
    Returns the result as a string. Raises ValueError for invalid expressions.
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string")

    expr = expression.strip()
    if not expr:
        raise ValueError("Empty expression")

    try:
        tokens = _tokenize(expr)
        rpn = _to_rpn(tokens)
        result = _eval_rpn(rpn)

        # Normalize output formatting: integers without decimal point, decimals trimmed
        # Handle negative zero
        if result == 0:
            return "0"

        # If it's an integer value, return as integer string
        if result == result.to_integral_value():
            return str(int(result))

        # Otherwise, fixed-point string without trailing zeros
        s = format(result, "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s if s else "0"
    except (InvalidOperation, DivisionByZero, ArithmeticError, ValueError) as e:
        raise ValueError("Invalid expression") from e


def _tokenize(expr: str) -> List[str]:
    """
    Convert the expression string into a list of tokens (numbers, operators, parentheses).
    Handles unary + and - by inserting a leading 0 (e.g., -x -> 0 - x).
    """
    tokens: List[str] = []
    i = 0
    n = len(expr)

    def is_operator_char(c: str) -> bool:
        return c in "+-*/"

    def is_digit_or_dot(c: str) -> bool:
        return c.isdigit() or c == "."

    prev_token: str | None = None  # Track previous non-space token

    while i < n:
        c = expr[i]

        if c.isspace():
            i += 1
            continue

        if c in "()+-*/":
            # Handle unary +/-
            if c in "+-" and (prev_token is None or prev_token in "+-*/("):
                # Insert a leading 0 to convert unary into binary
                tokens.append("0")
                tokens.append(c)
                prev_token = c
                i += 1
                continue

            tokens.append(c)
            prev_token = c
            i += 1
            continue

        # Number (integer or decimal)
        if is_digit_or_dot(c):
            start = i
            dot_count = 0
            while i < n and is_digit_or_dot(expr[i]):
                if expr[i] == ".":
                    dot_count += 1
                    if dot_count > 1:
                        raise ValueError("Invalid number format")
                i += 1

            num_str = expr[start:i]
            # Validate that it's not just '.' or empty
            if num_str == ".":
                raise ValueError("Invalid number format")
            tokens.append(num_str)
            prev_token = num_str
            continue

        # Any other character is invalid
        raise ValueError(f"Invalid character: {c}")

    return tokens


def _to_rpn(tokens: List[str]) -> List[Token]:
    """
    Convert list of tokens to Reverse Polish Notation using the shunting-yard algorithm.
    """
    output: List[Token] = []
    ops_stack: List[str] = []

    precedence = {"+": 1, "-": 1, "*": 2, "/": 2}
    left_assoc = {"+", "-", "*", "/"}

    for tok in tokens:
        if _is_number_token(tok):
            output.append(Decimal(tok))
        elif tok in precedence:
            while (
                ops_stack
                and ops_stack[-1] in precedence
                and (
                    (ops_stack[-1] in left_assoc and precedence[ops_stack[-1]] >= precedence[tok])
                )
            ):
                output.append(ops_stack.pop())
            ops_stack.append(tok)
        elif tok == "(":
            ops_stack.append(tok)
        elif tok == ")":
            while ops_stack and ops_stack[-1] != "(":
                output.append(ops_stack.pop())
            if not ops_stack or ops_stack[-1] != "(":
                raise ValueError("Mismatched parentheses")
            ops_stack.pop()  # Discard '('
        else:
            raise ValueError(f"Invalid token: {tok}")

    while ops_stack:
        op = ops_stack.pop()
        if op in ("(", ")"):
            raise ValueError("Mismatched parentheses")
        output.append(op)

    return output


def _eval_rpn(rpn: List[Token]) -> Decimal:
    """
    Evaluate an RPN expression using Decimal arithmetic.
    """
    stack: List[Decimal] = []

    for tok in rpn:
        if isinstance(tok, Decimal):
            stack.append(tok)
        else:
            # Operator
            if len(stack) < 2:
                raise ValueError("Insufficient operands")
            b = stack.pop()
            a = stack.pop()
            if tok == "+":
                stack.append(a + b)
            elif tok == "-":
                stack.append(a - b)
            elif tok == "*":
                stack.append(a * b)
            elif tok == "/":
                if b == 0:
                    raise DivisionByZero("division by zero")
                stack.append(a / b)
            else:
                raise ValueError(f"Unknown operator: {tok}")

    if len(stack) != 1:
        raise ValueError("Invalid expression")
    return stack[0]


def _is_number_token(tok: str) -> bool:
    # A number token is composed of digits and at most one dot, but we already validated on tokenize
    # We still do a quick check here.
    if not tok:
        return False
    if tok.count(".") > 1:
        return False
    return all(ch.isdigit() or ch == "." for ch in tok)
