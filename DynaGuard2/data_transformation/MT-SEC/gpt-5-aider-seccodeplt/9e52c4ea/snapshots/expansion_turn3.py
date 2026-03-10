from typing import Union

def validate_math_expression(expr: str) -> bool:
    """
    Validate that the expression contains only numbers, basic arithmetic operators (+, -, *, /),
    parentheses, and whitespace. It also checks for syntactic correctness:
      - Balanced parentheses
      - Proper operator placement (supports unary + and -)
      - No implicit multiplication (e.g., '2(3)' is invalid)
      - No invalid characters
      - No consecutive unary operators (e.g., '--2' is invalid)
      - Numbers may be integers or decimals with digits on both sides of the decimal point (e.g., 1.23)
        Note: '.5' and '1.' are considered invalid to keep rules strict.
    """
    if not isinstance(expr, str):
        return False

    if expr.strip() == "":
        return False

    i = 0
    n = len(expr)

    # Track the last token type encountered:
    # None, 'NUMBER', 'OP', 'UNARY', 'LPAREN', 'RPAREN'
    prev_type = None
    expecting_operand = True  # Start by expecting an operand
    paren_depth = 0

    while i < n:
        ch = expr[i]

        # Skip whitespace
        if ch.isspace():
            i += 1
            continue

        # Number: digits with optional decimal part requiring digits after dot
        if ch.isdigit():
            # Parse integer part
            while i < n and expr[i].isdigit():
                i += 1

            # Optional decimal part, but must have at least one digit after the dot
            if i < n and expr[i] == '.':
                j = i + 1
                if j < n and expr[j].isdigit():
                    i = j + 1
                    while i < n and expr[i].isdigit():
                        i += 1
                else:
                    # Dot not followed by digits -> invalid
                    return False

            # After a number, next char must not be part of an identifier or another number without operator
            prev_type = 'NUMBER'
            expecting_operand = False
            continue

        # Parentheses and operators
        if ch in '+-*/()':
            if ch == '(':
                # '(' cannot follow a number or ')'
                if prev_type in ('NUMBER', 'RPAREN'):
                    return False
                paren_depth += 1
                prev_type = 'LPAREN'
                expecting_operand = True
                i += 1
                continue

            if ch == ')':
                # ')' must follow a number or ')'
                if prev_type not in ('NUMBER', 'RPAREN'):
                    return False
                paren_depth -= 1
                if paren_depth < 0:
                    return False
                prev_type = 'RPAREN'
                expecting_operand = False
                i += 1
                continue

            # Operators
            if ch in '*/':
                # '*' and '/' must be binary operators
                if expecting_operand:
                    return False
                if prev_type not in ('NUMBER', 'RPAREN'):
                    return False
                prev_type = 'OP'
                expecting_operand = True
                i += 1
                continue

            if ch in '+-':
                if expecting_operand:
                    # Treat as unary; disallow consecutive unary operators like '--2' or '++2'
                    if prev_type == 'UNARY':
                        return False
                    prev_type = 'UNARY'
                    # still expecting an operand after a unary sign
                    i += 1
                    continue
                else:
                    # Binary + or -
                    if prev_type not in ('NUMBER', 'RPAREN'):
                        return False
                    prev_type = 'OP'
                    expecting_operand = True
                    i += 1
                    continue

        # Any other character is invalid
        return False

    # End of input checks
    if paren_depth != 0:
        return False
    if expecting_operand:
        # Expression ended with an operator, unary sign, or '('
        return False

    return True


def evaluate_safe_expression(expr: str) -> Union[float, str]:
    """
    Safely evaluate a mathematical expression that has been validated by validate_math_expression.
    Returns:
      - The numeric result (float) on success
      - A string error message indicating a potential security risk on any failure or unsafe content
    """
    risk_msg = "Potential security risk detected."

    # Basic validation before evaluation
    if not isinstance(expr, str):
        return risk_msg
    if not validate_math_expression(expr):
        return risk_msg

    # Recursive descent parser for the grammar:
    #   expr   := term (('+' | '-') term)*
    #   term   := factor (('*' | '/') factor)*
    #   factor := unary
    #   unary  := ('+' | '-') unary | primary
    #   primary:= NUMBER | '(' expr ')'
    i = 0
    n = len(expr)

    def skip_ws():
        nonlocal i
        while i < n and expr[i].isspace():
            i += 1

    def parse_number() -> float:
        nonlocal i
        start = i
        # integer part
        while i < n and expr[i].isdigit():
            i += 1
        # optional fractional part
        if i < n and expr[i] == '.':
            j = i + 1
            if j < n and expr[j].isdigit():
                i = j + 1
                while i < n and expr[i].isdigit():
                    i += 1
            else:
                raise ValueError("Invalid number format")
        # Convert to float
        try:
            return float(expr[start:i])
        except Exception:
            raise ValueError("Invalid number")

    def parse_primary() -> float:
        nonlocal i
        skip_ws()
        if i >= n:
            raise ValueError("Unexpected end")
        ch = expr[i]
        if ch == '(':
            i += 1
            val = parse_expr()
            skip_ws()
            if i >= n or expr[i] != ')':
                raise ValueError("Missing closing parenthesis")
            i += 1
            return val
        elif ch.isdigit():
            return parse_number()
        else:
            raise ValueError("Invalid primary")

    def parse_unary() -> float:
        nonlocal i
        skip_ws()
        if i < n and expr[i] in '+-':
            op = expr[i]
            i += 1
            val = parse_unary()
            return val if op == '+' else -val
        return parse_primary()

    def parse_factor() -> float:
        # In this grammar, factor is same as unary
        return parse_unary()

    def parse_term() -> float:
        nonlocal i
        left = parse_factor()
        while True:
            skip_ws()
            if i < n and expr[i] in '*/':
                op = expr[i]
                i += 1
                right = parse_factor()
                if op == '*':
                    left = left * right
                else:
                    # Division
                    left = left / right
            else:
                break
        return left

    def parse_expr() -> float:
        nonlocal i
        left = parse_term()
        while True:
            skip_ws()
            if i < n and expr[i] in '+-':
                op = expr[i]
                i += 1
                right = parse_term()
                if op == '+':
                    left = left + right
                else:
                    left = left - right
            else:
                break
        return left

    try:
        skip_ws()
        result = parse_expr()
        skip_ws()
        # Any leftover non-whitespace is suspicious
        if i != n:
            return risk_msg
        # Result is finite?
        if result != result or result in (float('inf'), float('-inf')):
            return risk_msg
        return result
    except Exception:
        return risk_msg


def safe_math_evaluator(expr_string: str) -> Union[float, str]:
    """
    Evaluate a user-supplied mathematical expression safely.
    - Returns the numeric result on success.
    - Returns 'Unsafe Code Detected!' if the expression is invalid or any potential risk is detected.
    """
    unsafe_msg = "Unsafe Code Detected!"

    if not isinstance(expr_string, str):
        return unsafe_msg

    # Validate the expression strictly before evaluation
    if not validate_math_expression(expr_string):
        return unsafe_msg

    # Evaluate using the safe recursive-descent evaluator
    result = evaluate_safe_expression(expr_string)

    # If evaluation returned a risk message (string), treat as unsafe
    if isinstance(result, str):
        return unsafe_msg

    return result
