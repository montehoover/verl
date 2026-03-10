def is_valid_expression(expr: str) -> bool:
    """
    Check if the given string is a well-formed arithmetic expression using only:
    - numbers (integers or decimals)
    - unary plus/minus
    - binary operators: +, -, *, /
    - parentheses
    - optional whitespace

    Rules:
    - Balanced parentheses are required.
    - No invalid characters.
    - Operators must be placed correctly (no missing operands).
    - Supports decimals like "3.14" and ".5" (but not "1." without digits after the dot).
    - Allows unary + and - (e.g., "-3", "1+-2", "(-5)").
    - Does not evaluate the expression; only validates syntax/format.

    Returns True if valid, False otherwise.
    """
    if not isinstance(expr, str):
        return False

    s = expr.strip()
    if not s:
        return False

    # Token types
    TT_NUMBER = "NUMBER"
    TT_PLUS = "PLUS"
    TT_MINUS = "MINUS"
    TT_MUL = "MUL"
    TT_DIV = "DIV"
    TT_LPAREN = "LPAREN"
    TT_RPAREN = "RPAREN"
    TT_EOF = "EOF"
    TT_INVALID = "INVALID"

    i = 0
    n = len(s)
    tok_type = None
    tok_val = None

    def is_digit(c: str) -> bool:
        return '0' <= c <= '9'

    def skip_ws():
        nonlocal i
        while i < n and s[i].isspace():
            i += 1

    def next_token():
        nonlocal i, tok_type, tok_val
        skip_ws()
        if i >= n:
            tok_type, tok_val = TT_EOF, None
            return

        c = s[i]

        # Numbers: digits+ ( '.' digits+ )?  OR  '.' digits+
        if is_digit(c):
            start = i
            while i < n and is_digit(s[i]):
                i += 1
            if i < n and s[i] == '.':
                j = i + 1
                if j < n and is_digit(s[j]):
                    i = j + 1
                    while i < n and is_digit(s[i]):
                        i += 1
                else:
                    # Disallow trailing '.' without digits after
                    tok_type, tok_val = TT_INVALID, None
                    return
            tok_type, tok_val = TT_NUMBER, s[start:i]
            return
        elif c == '.':
            j = i + 1
            if j < n and is_digit(s[j]):
                start = i
                i = j + 1
                while i < n and is_digit(s[i]):
                    i += 1
                tok_type, tok_val = TT_NUMBER, s[start:i]
                return
            else:
                tok_type, tok_val = TT_INVALID, None
                return

        # Operators and parentheses
        if c == '+':
            i += 1
            tok_type, tok_val = TT_PLUS, '+'
            return
        if c == '-':
            i += 1
            tok_type, tok_val = TT_MINUS, '-'
            return
        if c == '*':
            i += 1
            tok_type, tok_val = TT_MUL, '*'
            return
        if c == '/':
            i += 1
            tok_type, tok_val = TT_DIV, '/'
            return
        if c == '(':
            i += 1
            tok_type, tok_val = TT_LPAREN, '('
            return
        if c == ')':
            i += 1
            tok_type, tok_val = TT_RPAREN, ')'
            return

        tok_type, tok_val = TT_INVALID, None

    def advance():
        next_token()

    def accept(t):
        nonlocal tok_type
        if tok_type == t:
            advance()
            return True
        return False

    def parse_expression() -> bool:
        # expr := term (('+' | '-') term)*
        if not parse_term():
            return False
        while tok_type in (TT_PLUS, TT_MINUS):
            advance()
            if not parse_term():
                return False
        return True

    def parse_term() -> bool:
        # term := factor (('*' | '/') factor)*
        if not parse_factor():
            return False
        while tok_type in (TT_MUL, TT_DIV):
            advance()
            if not parse_factor():
                return False
        return True

    def parse_factor() -> bool:
        # factor := NUMBER | '(' expr ')' | ('+'|'-') factor   (unary)
        if tok_type in (TT_PLUS, TT_MINUS):
            # unary plus/minus
            advance()
            return parse_factor()
        if tok_type == TT_NUMBER:
            advance()
            return True
        if tok_type == TT_LPAREN:
            advance()
            if not parse_expression():
                return False
            if not accept(TT_RPAREN):
                return False
            return True
        return False

    # Kick off
    advance()
    if tok_type == TT_EOF:
        # Empty after trimming is already handled, but guard anyway
        return False
    if tok_type == TT_INVALID:
        return False

    ok = parse_expression()
    if not ok:
        return False
    # After a valid expression, we must be at EOF
    return tok_type == TT_EOF


def calculate_expression(expr: str):
    """
    Safely compute the result of a mathematical expression using only:
    - numbers (integers or decimals)
    - unary plus/minus
    - binary operators: +, -, *, /
    - parentheses
    - optional whitespace

    Returns:
      - float result if computation is successful
      - str error message if the expression is invalid/unsafe or computation fails
    """
    # Validate input and syntax first
    if not isinstance(expr, str):
        return "Invalid input type."
    s = expr.strip()
    if not s:
        return "Empty expression."
    if not is_valid_expression(s):
        return "Invalid or unsafe expression."

    # Token types for evaluator
    TT_NUMBER = "NUMBER"
    TT_PLUS = "PLUS"
    TT_MINUS = "MINUS"
    TT_MUL = "MUL"
    TT_DIV = "DIV"
    TT_LPAREN = "LPAREN"
    TT_RPAREN = "RPAREN"
    TT_EOF = "EOF"
    TT_INVALID = "INVALID"

    i = 0
    n = len(s)
    tok_type = None
    tok_val = None  # float for numbers, otherwise symbol

    def is_digit(c: str) -> bool:
        return '0' <= c <= '9'

    def skip_ws():
        nonlocal i
        while i < n and s[i].isspace():
            i += 1

    def next_token():
        nonlocal i, tok_type, tok_val
        skip_ws()
        if i >= n:
            tok_type, tok_val = TT_EOF, None
            return

        c = s[i]

        # Numbers: digits+ ( '.' digits+ )?  OR  '.' digits+
        if is_digit(c):
            start = i
            while i < n and is_digit(s[i]):
                i += 1
            if i < n and s[i] == '.':
                j = i + 1
                if j < n and is_digit(s[j]):
                    i = j + 1
                    while i < n and is_digit(s[i]):
                        i += 1
                else:
                    tok_type, tok_val = TT_INVALID, None
                    return
            num_str = s[start:i]
            try:
                tok_type, tok_val = TT_NUMBER, float(num_str)
            except Exception:
                tok_type, tok_val = TT_INVALID, None
            return
        elif c == '.':
            j = i + 1
            if j < n and is_digit(s[j]):
                start = i
                i = j + 1
                while i < n and is_digit(s[i]):
                    i += 1
                num_str = s[start:i]
                try:
                    tok_type, tok_val = TT_NUMBER, float(num_str)
                except Exception:
                    tok_type, tok_val = TT_INVALID, None
                return
            else:
                tok_type, tok_val = TT_INVALID, None
                return

        # Operators and parentheses
        if c == '+':
            i += 1
            tok_type, tok_val = TT_PLUS, '+'
            return
        if c == '-':
            i += 1
            tok_type, tok_val = TT_MINUS, '-'
            return
        if c == '*':
            i += 1
            tok_type, tok_val = TT_MUL, '*'
            return
        if c == '/':
            i += 1
            tok_type, tok_val = TT_DIV, '/'
            return
        if c == '(':
            i += 1
            tok_type, tok_val = TT_LPAREN, '('
            return
        if c == ')':
            i += 1
            tok_type, tok_val = TT_RPAREN, ')'
            return

        tok_type, tok_val = TT_INVALID, None

    def advance():
        next_token()

    def accept(t):
        nonlocal tok_type
        if tok_type == t:
            advance()
            return True
        return False

    def expect(t):
        if not accept(t):
            raise ValueError("Invalid or unsafe expression.")

    def parse_expression():
        # expr := term (('+' | '-') term)*
        val = parse_term()
        while tok_type in (TT_PLUS, TT_MINUS):
            op = tok_type
            advance()
            rhs = parse_term()
            if op == TT_PLUS:
                val = val + rhs
            else:
                val = val - rhs
        return val

    def parse_term():
        # term := factor (('*' | '/') factor)*
        val = parse_factor()
        while tok_type in (TT_MUL, TT_DIV):
            op = tok_type
            advance()
            rhs = parse_factor()
            if op == TT_MUL:
                val = val * rhs
            else:
                if rhs == 0.0:
                    raise ZeroDivisionError("Division by zero.")
                val = val / rhs
        return val

    def parse_factor():
        # factor := NUMBER | '(' expr ')' | ('+'|'-') factor   (unary)
        if tok_type in (TT_PLUS, TT_MINUS):
            op = tok_type
            advance()
            val = parse_factor()
            return val if op == TT_PLUS else -val

        if tok_type == TT_NUMBER:
            val = tok_val
            advance()
            return val

        if tok_type == TT_LPAREN:
            advance()
            val = parse_expression()
            expect(TT_RPAREN)
            return val

        raise ValueError("Invalid or unsafe expression.")

    try:
        advance()
        if tok_type in (TT_EOF, TT_INVALID):
            return "Invalid or unsafe expression."
        result = parse_expression()
        if tok_type != TT_EOF:
            return "Invalid or unsafe expression."
        return result
    except ZeroDivisionError:
        return "Division by zero."
    except OverflowError:
        return "Computation overflow."
    except Exception:
        return "Invalid or unsafe expression."
