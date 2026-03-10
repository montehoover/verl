import ast

def evaluate_expression(expression):
    """
    Evaluate a simple expression:
      - Integer arithmetic with + and - operators (supports unary + and - on numbers).
      - String concatenation with + between quoted strings ('...' or "...").
    Returns:
      - int result for numeric expressions
      - str result for string concatenations
      - 'Invalid Expression' string if invalid
    """
    if not isinstance(expression, str):
        return 'Invalid Expression'

    s = expression
    n = len(s)

    def skip_spaces(i):
        while i < n and s[i].isspace():
            i += 1
        return i

    def parse_signed_int(i):
        i = skip_spaces(i)
        if i >= n:
            return None, i

        sign = 1
        if s[i] == '+' or s[i] == '-':
            if s[i] == '-':
                sign = -1
            i += 1

        start_digits = i
        while i < n and s[i].isdigit():
            i += 1

        if i == start_digits:
            return None, start_digits  # no digits after optional sign

        value = int(s[start_digits:i]) * sign
        return value, i

    def parse_string_literal(i):
        i = skip_spaces(i)
        if i >= n:
            return None, i
        if s[i] not in ("'", '"'):
            return None, i

        quote = s[i]
        i += 1
        buf = []
        while i < n:
            ch = s[i]
            if ch == '\\':
                i += 1
                if i >= n:
                    return None, i
                esc = s[i]
                # Support common escapes and escaped quotes/backslash
                if esc == 'n':
                    buf.append('\n')
                elif esc == 't':
                    buf.append('\t')
                elif esc == 'r':
                    buf.append('\r')
                elif esc == '\\':
                    buf.append('\\')
                elif esc == "'":
                    buf.append("'")
                elif esc == '"':
                    buf.append('"')
                else:
                    # Unknown escape, treat as literal character
                    buf.append(esc)
                i += 1
            elif ch == quote:
                i += 1
                return ''.join(buf), i
            else:
                buf.append(ch)
                i += 1

        # If we exit the loop, no closing quote was found
        return None, i

    # Attempt to parse as numeric expression
    def try_numeric():
        i = skip_spaces(0)
        first_val, i2 = parse_signed_int(i)
        if first_val is None:
            return None
        result = first_val
        i = i2

        while True:
            i = skip_spaces(i)
            if i >= n:
                break
            if s[i] not in ('+', '-'):
                return None
            op = s[i]
            i += 1

            next_val, i2 = parse_signed_int(i)
            if next_val is None:
                return None
            if op == '+':
                result += next_val
            else:
                result -= next_val
            i = i2

        i = skip_spaces(i)
        if i != n:
            return None
        return result

    # Attempt to parse as string concatenation expression (only + allowed)
    def try_string():
        i = skip_spaces(0)
        first_str, i2 = parse_string_literal(i)
        if first_str is None:
            return None
        result = first_str
        i = i2

        while True:
            i = skip_spaces(i)
            if i >= n:
                break
            if s[i] != '+':
                return None
            i += 1

            next_str, i2 = parse_string_literal(i)
            if next_str is None:
                return None
            result += next_str
            i = i2

        i = skip_spaces(i)
        if i != n:
            return None
        return result

    numeric_result = try_numeric()
    if numeric_result is not None:
        return numeric_result

    string_result = try_string()
    if string_result is not None:
        return string_result

    return 'Invalid Expression'


def parse_user_script(script_str):
    """
    Safely evaluate a user-provided Python script string that should contain only
    basic arithmetic or string operations. On any error or detection of unsafe
    constructs, returns 'Execution Blocked!'.
    """
    if not isinstance(script_str, str):
        return 'Execution Blocked!'
    if script_str.strip() == '':
        return 'Execution Blocked!'

    try:
        node = ast.parse(script_str, mode='eval')  # only allow a single expression
    except Exception:
        return 'Execution Blocked!'

    try:
        return _safe_eval_ast(node)
    except Exception:
        return 'Execution Blocked!'


def _safe_eval_ast(node):
    """
    Evaluate a restricted subset of Python AST:
      - numeric constants (int, float) and string constants
      - binary operations: +, -, *, /, //, % on numbers; + on strings
      - unary operations: +, - on numbers
    Disallows all names, calls, attributes, subscripts, f-strings, and other nodes.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body)

    if isinstance(node, ast.Constant):
        # Allow only int, float, str (exclude bool/None/etc.)
        if type(node.value) in (int, float, str):
            return node.value
        raise ValueError('disallowed constant')

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            if isinstance(operand, (int, float)):
                return +operand
            raise ValueError('type mismatch for unary +')
        if isinstance(node.op, ast.USub):
            if isinstance(operand, (int, float)):
                return -operand
            raise ValueError('type mismatch for unary -')
        raise ValueError('unary operator not allowed')

    if isinstance(node, ast.BinOp):
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        op = node.op

        if isinstance(op, ast.Add):
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return left + right
            if isinstance(left, str) and isinstance(right, str):
                return left + right
            raise ValueError('type mismatch for +')

        if isinstance(op, (ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if isinstance(op, ast.Sub):
                    return left - right
                if isinstance(op, ast.Mult):
                    return left * right
                if isinstance(op, ast.Div):
                    return left / right
                if isinstance(op, ast.FloorDiv):
                    return left // right
                if isinstance(op, ast.Mod):
                    return left % right
            raise ValueError('numeric operator type mismatch')

        raise ValueError('binary operator not allowed')

    # Explicitly reject potentially unsafe or out-of-scope nodes
    disallowed = (
        ast.Call, ast.Name, ast.Attribute, ast.Subscript, ast.Slice,
        ast.Dict, ast.List, ast.Tuple, ast.Set, ast.ListComp, ast.SetComp,
        ast.DictComp, ast.GeneratorExp, ast.Lambda, ast.IfExp, ast.Compare,
        ast.BoolOp, ast.AugAssign, ast.Assign, ast.NamedExpr, ast.JoinedStr,
        ast.FormattedValue, ast.Yield, ast.YieldFrom, ast.Await
    )
    if isinstance(node, disallowed):
        raise ValueError('disallowed node')

    # Any other node is disallowed
    raise ValueError('unknown node')
