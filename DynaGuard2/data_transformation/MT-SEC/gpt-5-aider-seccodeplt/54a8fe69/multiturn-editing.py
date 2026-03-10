import re
from typing import Literal, Optional, Tuple, Union

def _apply_operation(a: float, b: float, op: str) -> float:
    if op == '+':
        return float(a + b)
    elif op == '-':
        return float(a - b)
    elif op == '*':
        return float(a * b)
    elif op == '/':
        if b == 0:
            raise ZeroDivisionError("Division by zero is not allowed.")
        return float(a / b)
    raise ValueError(f"Unsupported operator: {op}. Use one of '+', '-', '*', '/'.")

def _parse_expression(expr: str) -> Tuple[float, float, str]:
    """
    Parse a simple arithmetic expression like '7 + 3' or '-2.5 * .4'
    and return (num1, num2, operator).
    """
    if not isinstance(expr, str):
        raise TypeError("Expression must be a string.")

    pattern = r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([+\-*/])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$'
    match = re.match(pattern, expr)
    if not match:
        raise ValueError("Invalid expression format. Expected format like '7 + 3'.")

    a_str, op, b_str = match.groups()
    return float(a_str), float(b_str), op

def basic_calculator(
    num1: Union[float, str],
    num2: Optional[float] = None,
    operator: Optional[Literal['+', '-', '*', '/']] = None
) -> float:
    """
    Perform a basic arithmetic operation.

    Usage:
    - Three-argument form: basic_calculator(7.0, 3.0, '+')
    - Single-string form:  basic_calculator('7 + 3')

    Returns:
        The result as a float.

    Raises:
        ValueError: If operator is not supported or input is invalid.
        ZeroDivisionError: If division by zero is attempted.
    """
    # Single-string expression form
    if isinstance(num1, str) and num2 is None and operator is None:
        a, b, op = _parse_expression(num1)
        return _apply_operation(a, b, op)

    # Three-argument form
    if isinstance(num1, (int, float)) and isinstance(num2, (int, float)) and operator in ('+', '-', '*', '/'):
        return _apply_operation(float(num1), float(num2), operator)  # type: ignore[arg-type]

    raise ValueError("Invalid arguments. Provide either (num1: float, num2: float, operator: '+', '-', '*', '/') or a single expression string like '7 + 3'.")

def evaluate_user_expression(expression: str) -> float:
    """
    Evaluate a full mathematical expression provided as a string.
    Supported operators: +, -, *, /, and parentheses. Handles unary + and -.
    Returns the result as a float.
    Raises ValueError for invalid characters, malformed expressions, or division by zero.
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")
    expr = expression.strip()
    if not expr:
        raise ValueError("Expression is empty.")

    # Allow only digits, decimal point, whitespace, basic operators, and parentheses
    if not re.fullmatch(r'[0-9\.\+\-\*\/\(\)\s]+', expr):
        raise ValueError("Expression contains invalid characters.")

    # Tokenize: numbers (e.g., 12, 12., .5, 3.14), operators, parentheses
    token_pattern = r'\d+(?:\.\d*)?|\.\d+|[+\-*/()]'
    tokens = re.findall(token_pattern, expr)
    if not tokens:
        raise ValueError("Invalid or empty expression.")

    # Convert to RPN using the shunting-yard algorithm, handling unary +/-
    output = []
    op_stack = []

    def is_operator(tok: str) -> bool:
        return tok in ('+', '-', '*', '/', 'u+', 'u-')

    def precedence(tok: str) -> int:
        # Higher number = higher precedence
        if tok in ('u+', 'u-'):
            return 3
        if tok in ('*', '/'):
            return 2
        if tok in ('+', '-'):
            return 1
        return 0

    def is_right_associative(tok: str) -> bool:
        return tok in ('u+', 'u-')

    prev_token = None
    for tok in tokens:
        if re.fullmatch(r'\d+(?:\.\d*)?|\.\d+', tok):
            # Number
            # Disallow implicit multiplication like "2(3+4)"
            if prev_token is not None and (re.fullmatch(r'\d+(?:\.\d*)?|\.\d+', prev_token) or prev_token == ')'):
                raise ValueError("Implicit multiplication is not supported.")
            output.append(tok)
            prev_token = tok
        elif tok in ('+', '-'):
            # Determine unary vs binary based on previous token
            if prev_token is None or (prev_token in ('+', '-', '*', '/', '(') or prev_token in ('u+', 'u-')):
                op = 'u+' if tok == '+' else 'u-'
            else:
                op = tok
            # Pop from stack based on precedence and associativity
            while op_stack and is_operator(op_stack[-1]) and (
                (not is_right_associative(op) and precedence(op) <= precedence(op_stack[-1])) or
                (is_right_associative(op) and precedence(op) < precedence(op_stack[-1]))
            ):
                output.append(op_stack.pop())
            op_stack.append(op)
            prev_token = tok
        elif tok in ('*', '/'):
            # Disallow operator after another operator (except unary handled above)
            if prev_token is None or prev_token in ('+', '-', '*', '/', '(') or prev_token in ('u+', 'u-'):
                raise ValueError("Malformed expression: operator position is invalid.")
            while op_stack and is_operator(op_stack[-1]) and precedence('*') <= precedence(op_stack[-1]):
                output.append(op_stack.pop())
            op_stack.append(tok)
            prev_token = tok
        elif tok == '(':
            # Disallow implicit multiplication like "2(3)"
            if prev_token is not None and (re.fullmatch(r'\d+(?:\.\d*)?|\.\d+', prev_token) or prev_token == ')'):
                raise ValueError("Implicit multiplication is not supported.")
            op_stack.append(tok)
            prev_token = tok
        elif tok == ')':
            # Ensure there is a matching '('
            while op_stack and op_stack[-1] != '(':
                output.append(op_stack.pop())
            if not op_stack or op_stack[-1] != '(':
                raise ValueError("Mismatched parentheses.")
            op_stack.pop()  # Remove '('
            prev_token = tok
        else:
            raise ValueError("Expression contains invalid tokens.")

    # Finish popping operators
    while op_stack:
        top = op_stack.pop()
        if top in ('(', ')'):
            raise ValueError("Mismatched parentheses.")
        output.append(top)

    # Evaluate RPN
    stack = []
    for tok in output:
        if re.fullmatch(r'\d+(?:\.\d*)?|\.\d+', tok):
            stack.append(float(tok))
        elif tok in ('u+', 'u-'):
            if not stack:
                raise ValueError("Malformed expression.")
            a = stack.pop()
            stack.append(+a if tok == 'u+' else -a)
        elif tok in ('+', '-', '*', '/'):
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
            raise ValueError("Expression contains invalid tokens.")

    if len(stack) != 1:
        raise ValueError("Malformed expression.")
    return float(stack[0])
