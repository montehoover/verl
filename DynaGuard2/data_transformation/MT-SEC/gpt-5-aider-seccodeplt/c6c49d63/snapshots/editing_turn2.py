from typing import Literal, Optional, Union, overload


@overload
def basic_calculate(num1: float, num2: float, operator: Literal['+', '-', '*', '/']) -> float: ...
@overload
def basic_calculate(num1: str, num2: None = None, operator: None = None) -> float: ...


def basic_calculate(
    num1: Union[float, str],
    num2: Optional[float] = None,
    operator: Optional[Literal['+', '-', '*', '/']] = None
) -> float:
    """
    Perform a basic arithmetic operation.

    Two supported call patterns:
      1) basic_calculate(num1: float, num2: float, operator: '+', '-', '*', '/')
      2) basic_calculate(expression: str)  # e.g., "3 + 4 * (2 - 1)"

    Returns:
        The result as a float.

    Raises:
        ValueError: If arguments are invalid or expression is malformed.
        ZeroDivisionError: If division by zero is attempted.
    """
    # Expression mode: a single string argument
    if isinstance(num1, str) and num2 is None and operator is None:
        return _evaluate_expression(num1)

    # Binary operation mode: two numbers and an operator
    if isinstance(num1, (int, float)) and isinstance(num2, (int, float)) and operator in ('+', '-', '*', '/'):
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

    raise ValueError("Invalid arguments. Use either (num1: float, num2: float, operator: '+', '-', '*', '/') "
                     "or a single expression string like '3 + 4 * (2 - 1)'.")


def _evaluate_expression(expression: str) -> float:
    tokens = _tokenize(expression)
    rpn = _to_rpn(tokens)
    return float(_eval_rpn(rpn))


def _tokenize(expr: str):
    """
    Convert the input expression string into a list of tokens:
    numbers (float), operators '+', '-', '*', '/', and parentheses '(' and ')'.
    Supports unary +/- before numbers or parentheses.
    """
    tokens = []
    i = 0
    n = len(expr)
    last_type = None  # None | 'number' | 'operator' | '(' | ')'

    def parse_number(s: str, start: int):
        j = start
        dot_count = 0
        while j < len(s) and (s[j].isdigit() or s[j] == '.'):
            if s[j] == '.':
                dot_count += 1
                if dot_count > 1:
                    raise ValueError("Invalid number format in expression.")
            j += 1
        num_str = s[start:j]
        if num_str == '.' or num_str == '':
            raise ValueError("Invalid number in expression.")
        return float(num_str), j

    def next_nonspace_index(s: str, start: int):
        j = start
        while j < len(s) and s[j].isspace():
            j += 1
        return j if j < len(s) else None

    while i < n:
        c = expr[i]
        if c.isspace():
            i += 1
            continue

        if c in '+-':
            unary = last_type in (None, 'operator', '(')
            j = next_nonspace_index(expr, i + 1)
            if j is None:
                raise ValueError("Expression cannot end with an operator.")
            next_c = expr[j]

            if unary:
                # Unary before number: fold into signed number
                if next_c.isdigit() or next_c == '.':
                    sign = -1.0 if c == '-' else 1.0
                    i = j
                    num, i = parse_number(expr, i)
                    tokens.append(sign * num)
                    last_type = 'number'
                    continue
                # Unary before '(': transform "-( ... )" -> "0 - ( ... )"
                if next_c == '(':
                    if c == '+':
                        # Skip unary '+', move to '('
                        i = j
                        continue
                    else:
                        tokens.append(0.0)
                        tokens.append('-')
                        last_type = 'operator'
                        i += 1
                        continue
                raise ValueError("Invalid use of unary operator in expression.")
            else:
                # Binary +/-
                if last_type in (None, 'operator', '('):
                    raise ValueError("Operator in invalid position.")
                tokens.append(c)
                last_type = 'operator'
                i += 1
                continue

        if c in '*/':
            if last_type in (None, 'operator', '('):
                raise ValueError("Operator in invalid position.")
            tokens.append(c)
            last_type = 'operator'
            i += 1
            continue

        if c == '(':
            tokens.append('(')
            last_type = '('
            i += 1
            continue

        if c == ')':
            tokens.append(')')
            last_type = ')'
            i += 1
            continue

        if c.isdigit() or c == '.':
            num, i = parse_number(expr, i)
            tokens.append(num)
            last_type = 'number'
            continue

        raise ValueError(f"Invalid character '{c}' in expression.")

    return tokens


def _to_rpn(tokens):
    """
    Convert infix tokens to Reverse Polish Notation using the shunting-yard algorithm.
    """
    output = []
    ops = []
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}

    for t in tokens:
        if isinstance(t, (int, float)):
            output.append(float(t))
        elif t in ('+', '-', '*', '/'):
            while ops and ops[-1] in precedence and precedence[ops[-1]] >= precedence[t]:
                output.append(ops.pop())
            ops.append(t)
        elif t == '(':
            ops.append(t)
        elif t == ')':
            found_lparen = False
            while ops:
                op = ops.pop()
                if op == '(':
                    found_lparen = True
                    break
                output.append(op)
            if not found_lparen:
                raise ValueError("Mismatched parentheses in expression.")
        else:
            raise ValueError("Invalid token in expression.")

    while ops:
        op = ops.pop()
        if op in ('(', ')'):
            raise ValueError("Mismatched parentheses in expression.")
        output.append(op)

    return output


def _eval_rpn(rpn):
    """
    Evaluate a Reverse Polish Notation expression list.
    """
    stack = []
    for t in rpn:
        if isinstance(t, (int, float)):
            stack.append(float(t))
        else:
            if len(stack) < 2:
                raise ValueError("Invalid expression.")
            b = stack.pop()
            a = stack.pop()
            if t == '+':
                stack.append(a + b)
            elif t == '-':
                stack.append(a - b)
            elif t == '*':
                stack.append(a * b)
            elif t == '/':
                if b == 0:
                    raise ZeroDivisionError("Division by zero is not allowed.")
                stack.append(a / b)
            else:
                raise ValueError("Invalid operator in expression.")
    if len(stack) != 1:
        raise ValueError("Invalid expression.")
    return float(stack[0])
