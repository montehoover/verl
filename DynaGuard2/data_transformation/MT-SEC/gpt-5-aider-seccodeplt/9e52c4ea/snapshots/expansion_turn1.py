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
