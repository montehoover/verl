def evaluate_expression(expr: str) -> str:
    """
    Evaluates a simple arithmetic expression containing integers with +, -, *, and / operators.
    - Respects operator precedence: * and / before + and -.
    - Supports optional unary + or - directly in front of a number (e.g., "-5 * 3", "1+-2").
    - Division must be exact (no remainder). If a division would produce a non-integer or divide by zero, it's invalid.
    Returns the result as a string, or 'Invalid Expression!' if the input is invalid.
    Examples:
      "1 + 2 - 3" -> "0"
      "-5 + 10"   -> "5"
      "1+-2"      -> "-1"
      "5 * 3"     -> "15"
      "10 / 2"    -> "5"
    """
    try:
        if not isinstance(expr, str):
            return 'Invalid Expression!'

        i = 0
        n = len(expr)

        def skip_ws():
            nonlocal i
            while i < n and expr[i].isspace():
                i += 1

        def parse_number() -> int:
            nonlocal i
            if i >= n or not expr[i].isdigit():
                raise ValueError('Expected number')
            value = 0
            while i < n and expr[i].isdigit():
                value = value * 10 + (ord(expr[i]) - 48)
                i += 1
            return value

        def parse_factor() -> int:
            nonlocal i
            skip_ws()
            if i >= n:
                raise ValueError('Unexpected end, expected factor')
            sign = 1
            if expr[i] in '+-':
                sign = -1 if expr[i] == '-' else 1
                i += 1
            if i >= n or not expr[i].isdigit():
                raise ValueError('Expected number after unary sign')
            value = parse_number()
            return sign * value

        def parse_term() -> int:
            nonlocal i
            left = parse_factor()
            while True:
                skip_ws()
                if i >= n:
                    break
                ch = expr[i]
                if ch == '*' or ch == '/':
                    i += 1
                    right = parse_factor()
                    if ch == '*':
                        left = left * right
                    else:
                        # Division: must be exact and non-zero divisor
                        if right == 0:
                            raise ValueError('Division by zero')
                        if left % right != 0:
                            raise ValueError('Non-integer division result')
                        left = left // right
                else:
                    break
            return left

        def parse_expression() -> int:
            nonlocal i
            left = parse_term()
            while True:
                skip_ws()
                if i >= n:
                    break
                ch = expr[i]
                if ch == '+' or ch == '-':
                    i += 1
                    right = parse_term()
                    if ch == '+':
                        left += right
                    else:
                        left -= right
                else:
                    break
            return left

        skip_ws()
        if i >= n:
            return 'Invalid Expression!'

        result = parse_expression()

        skip_ws()
        if i != n:
            return 'Invalid Expression!'

        return str(result)
    except Exception:
        return 'Invalid Expression!'
