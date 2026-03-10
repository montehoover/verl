def evaluate(expression):
    """
    Evaluate a simple arithmetic expression string supporting +, -, *, / and parentheses.
    Returns the numeric result, or the string 'Invalid Expression!' if the expression is invalid.
    """

    INVALID = 'Invalid Expression!'

    # Basic validation
    if not isinstance(expression, str):
        return INVALID
    s = expression.strip()
    if not s:
        return INVALID

    # Tokenization
    def tokenize(text):
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch.isspace():
                i += 1
                continue
            if ch in '+-*/()':
                tokens.append(ch)
                i += 1
                continue
            if ch.isdigit() or ch == '.':
                start = i
                has_dot = (ch == '.')
                i += 1
                while i < n:
                    c = text[i]
                    if c.isdigit():
                        i += 1
                        continue
                    if c == '.' and not has_dot:
                        has_dot = True
                        i += 1
                        continue
                    break
                num_str = text[start:i]
                # Reject lone dot or malformed numbers like "." or ".."
                if num_str == '.' or num_str.count('.') > 1:
                    raise ValueError('invalid number')
                try:
                    num_val = float(num_str)
                except Exception as e:
                    raise ValueError('invalid number') from e
                tokens.append(('NUMBER', num_val))
                continue
            # Invalid character
            raise ValueError('invalid character')
        return tokens

    try:
        tokens = tokenize(s)
    except Exception:
        return INVALID

    # Parser using recursive descent:
    # expr   := term (('+' | '-') term)*
    # term   := factor (('*' | '/') factor)*
    # factor := NUMBER | '(' expr ')' | ('+' | '-') factor
    idx = 0

    def peek():
        nonlocal idx
        if idx < len(tokens):
            return tokens[idx]
        return None

    def consume(expected=None):
        nonlocal idx
        tok = peek()
        if tok is None:
            raise ValueError('unexpected end')
        if expected is not None:
            if tok == expected:
                idx += 1
                return tok
            # For number type expectation
            if expected == 'NUMBER' and isinstance(tok, tuple) and tok[0] == 'NUMBER':
                idx += 1
                return tok
            raise ValueError('unexpected token')
        idx += 1
        return tok

    def parse_expression():
        value = parse_term()
        while True:
            tok = peek()
            if tok == '+' or tok == '-':
                op = consume()
                rhs = parse_term()
                if op == '+':
                    value = value + rhs
                else:
                    value = value - rhs
            else:
                break
        return value

    def parse_term():
        value = parse_factor()
        while True:
            tok = peek()
            if tok == '*' or tok == '/':
                op = consume()
                rhs = parse_factor()
                if op == '*':
                    value = value * rhs
                else:
                    # Division
                    if rhs == 0:
                        raise ZeroDivisionError('division by zero')
                    value = value / rhs
            else:
                break
        return value

    def parse_factor():
        tok = peek()
        # Unary operators
        if tok == '+':
            consume('+')
            return parse_factor()
        if tok == '-':
            consume('-')
            return -parse_factor()
        # Parenthesized expression
        if tok == '(':
            consume('(')
            val = parse_expression()
            if peek() != ')':
                raise ValueError('missing closing parenthesis')
            consume(')')
            return val
        # Number
        if isinstance(tok, tuple) and tok[0] == 'NUMBER':
            num = consume('NUMBER')[1]
            return num
        raise ValueError('expected number or "("')

    try:
        result = parse_expression()
        # Ensure all tokens consumed
        if peek() is not None:
            # Extra tokens remaining -> invalid
            return INVALID
        # Normalize -0.0 to 0.0
        if result == 0:
            result = 0.0
        # Return int if the result is integral
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result
    except Exception:
        return INVALID
