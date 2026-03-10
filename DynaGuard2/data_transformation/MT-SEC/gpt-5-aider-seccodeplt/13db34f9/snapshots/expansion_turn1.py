import operator

def evaluate_simple_expression(expression: str) -> float:
    """
    Evaluate a simple arithmetic expression containing +, -, *, / and parentheses.
    Returns the result as a float.
    Raises ValueError for invalid expressions (including division by zero).
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")
    s = expression.strip()
    if not s:
        raise ValueError("Empty expression.")

    # Tokenization
    tokens = []
    i = 0
    n = len(s)
    prev_type = None  # None, 'number', 'op', '(', ')'

    def parse_number(start_idx: int):
        i_local = start_idx
        saw_digit = False
        saw_dot = False
        while i_local < n and (s[i_local].isdigit() or s[i_local] == '.'):
            ch_local = s[i_local]
            if ch_local == '.':
                if saw_dot:
                    raise ValueError("Invalid number format.")
                saw_dot = True
            else:
                saw_digit = True
            i_local += 1
        if not saw_digit:
            raise ValueError("Invalid number format.")
        num_str = s[start_idx:i_local]
        try:
            return float(num_str), i_local
        except Exception as e:
            raise ValueError("Invalid number.") from e

    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit() or ch == '.':
            num, i = parse_number(i)
            tokens.append(num)
            prev_type = 'number'
            continue
        if ch in '+-':
            # Check for unary operator
            if prev_type in (None, 'op', '('):
                # Unary + or -
                next_char = s[i + 1] if (i + 1) < n else ''
                if next_char.isdigit() or next_char == '.':
                    sign = -1.0 if ch == '-' else 1.0
                    i += 1
                    num, i = parse_number(i)
                    tokens.append(sign * num)
                    prev_type = 'number'
                    continue
                elif next_char == '(':
                    # Transform unary +/- before '(' into 0 +/- ( ... )
                    tokens.append(0.0)
                    tokens.append('-' if ch == '-' else '+')
                    i += 1  # advance to '('; will be processed in next loop
                    prev_type = 'op'
                    continue
                else:
                    raise ValueError("Invalid use of unary operator.")
            # Binary + or -
            tokens.append(ch)
            i += 1
            prev_type = 'op'
            continue
        if ch in '*/':
            tokens.append(ch)
            i += 1
            prev_type = 'op'
            continue
        if ch == '(':
            tokens.append(ch)
            i += 1
            prev_type = '('
            continue
        if ch == ')':
            tokens.append(ch)
            i += 1
            prev_type = ')'
            continue
        # Invalid character
        raise ValueError(f"Invalid character in expression: {ch!r}")

    # Shunting-yard to convert to RPN
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    operators = set(precedence.keys())
    output = []
    op_stack = []

    for tok in tokens:
        if isinstance(tok, float):
            output.append(tok)
        elif tok in operators:
            while op_stack and op_stack[-1] in operators and precedence[op_stack[-1]] >= precedence[tok]:
                output.append(op_stack.pop())
            op_stack.append(tok)
        elif tok == '(':
            op_stack.append(tok)
        elif tok == ')':
            while op_stack and op_stack[-1] != '(':
                output.append(op_stack.pop())
            if not op_stack or op_stack[-1] != '(':
                raise ValueError("Mismatched parentheses.")
            op_stack.pop()  # remove '('
        else:
            raise ValueError("Invalid token encountered.")

    while op_stack:
        top = op_stack.pop()
        if top in ('(', ')'):
            raise ValueError("Mismatched parentheses.")
        output.append(top)

    # Evaluate RPN
    func_map = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
    }

    stack = []
    for tok in output:
        if isinstance(tok, float):
            stack.append(tok)
        elif tok in func_map:
            if len(stack) < 2:
                raise ValueError("Invalid expression.")
            b = stack.pop()
            a = stack.pop()
            if tok == '/' and b == 0.0:
                raise ValueError("Division by zero.")
            try:
                res = func_map[tok](a, b)
            except Exception as e:
                raise ValueError("Invalid operation.") from e
            stack.append(float(res))
        else:
            raise ValueError("Invalid token during evaluation.")

    if len(stack) != 1:
        raise ValueError("Invalid expression.")

    return float(stack[0])
