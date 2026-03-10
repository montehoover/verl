def evaluate_expression(expr: str) -> str:
    """
    Evaluates a simple arithmetic expression containing integers with + and - operators.
    Returns the result as a string, or 'Invalid Expression!' if the input is invalid.
    Examples:
      "1 + 2 - 3" -> "0"
      "-5 + 10"   -> "5"
      "1+-2"      -> "-1"
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

        skip_ws()
        if i >= n:
            return 'Invalid Expression!'

        total = None
        pending_op = None
        expect_number = True

        while True:
            if expect_number:
                skip_ws()
                if i >= n:
                    return 'Invalid Expression!'

                # Optional unary sign
                sign = 1
                if expr[i] in '+-':
                    sign = -1 if expr[i] == '-' else 1
                    i += 1

                # At least one digit must follow
                if i >= n or not expr[i].isdigit():
                    return 'Invalid Expression!'

                value = 0
                while i < n and expr[i].isdigit():
                    value = value * 10 + (ord(expr[i]) - 48)
                    i += 1
                value *= sign

                if total is None:
                    total = value
                else:
                    if pending_op == '+':
                        total += value
                    elif pending_op == '-':
                        total -= value
                    else:
                        return 'Invalid Expression!'

                expect_number = False
            else:
                skip_ws()
                if i >= n:
                    break

                ch = expr[i]
                if ch in '+-':
                    pending_op = ch
                    i += 1
                    expect_number = True
                else:
                    # Any non-operator character here is invalid
                    return 'Invalid Expression!'

        if total is None or expect_number:
            return 'Invalid Expression!'

        return str(total)
    except Exception:
        return 'Invalid Expression!'
