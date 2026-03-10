import operator
import re
import ast

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


def substitute_variables(expression: str, variables: dict) -> str:
    """
    Substitute variable identifiers in the expression with their numeric values.

    - Variables are identifiers matching [A-Za-z_][A-Za-z0-9_]*
    - Values in `variables` must be numeric (int, float) or strings representing numbers.
    - Raises ValueError if an identifier in the expression is not provided in `variables`
      or if a provided value is not numeric.
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")
    if not isinstance(variables, dict):
        raise ValueError("Variables must be a dict.")

    ident_pattern = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*\b')

    # Collect identifiers present in the expression
    identifiers = {m.group(0) for m in ident_pattern.finditer(expression)}
    if not identifiers:
        return expression

    # Ensure all identifiers are defined
    undefined = sorted(name for name in identifiers if name not in variables)
    if undefined:
        raise ValueError(f"Undefined variable(s): {', '.join(undefined)}")

    # Helper to ensure numeric substitution
    num_pattern = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)$')

    def value_to_numeric_string(val) -> str:
        if isinstance(val, bool):
            # Exclude bools which are ints subclass in Python
            raise ValueError("Boolean is not a valid numeric value.")
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, str):
            v = val.strip()
            if num_pattern.fullmatch(v):
                return v
        raise ValueError(f"Invalid value for variable substitution: {val!r}")

    def replacer(match: re.Match) -> str:
        name = match.group(0)
        return value_to_numeric_string(variables[name])

    return ident_pattern.sub(replacer, expression)


def evaluate_math_expression(formula: str, variables: dict) -> str:
    """
    Evaluate a mathematical expression with variables using a safe AST walker.
    - Supports +, -, *, /, unary +/-, and parentheses.
    - Variables are standard identifiers and must be provided in `variables`.
    - Variable values must be numeric (int, float) or numeric strings.
    - Returns the result as a string.
    - Raises ValueError for invalid expressions or operations (incl. division by zero).
    """
    if not isinstance(formula, str):
        raise ValueError("Expression must be a string.")
    if not isinstance(variables, dict):
        raise ValueError("Variables must be a dict.")

    text = formula.strip()
    if not text:
        raise ValueError("Empty expression.")

    # Prepare numeric coercion for variable values (aligned with substitute_variables)
    num_pattern = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$')

    def coerce_numeric(val) -> float:
        if isinstance(val, bool):
            raise ValueError("Boolean is not a valid numeric value.")
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            v = val.strip()
            if num_pattern.fullmatch(v):
                try:
                    return float(v)
                except Exception as e:
                    raise ValueError("Invalid numeric string.") from e
        raise ValueError(f"Invalid value for variable: {val!r}")

    # Safe evaluation via AST
    try:
        node = ast.parse(text, mode='eval')
    except SyntaxError as e:
        raise ValueError("Invalid expression.") from e

    bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def eval_node(n) -> float:
        if isinstance(n, ast.Expression):
            return eval_node(n.body)
        if isinstance(n, ast.BinOp):
            left = eval_node(n.left)
            right = eval_node(n.right)
            op_type = type(n.op)
            if op_type not in bin_ops:
                raise ValueError("Unsupported operator.")
            if op_type is ast.Div and right == 0.0:
                raise ValueError("Division by zero.")
            try:
                return float(bin_ops[op_type](left, right))
            except Exception as e:
                raise ValueError("Invalid operation.") from e
        if isinstance(n, ast.UnaryOp):
            op_type = type(n.op)
            if op_type not in unary_ops:
                raise ValueError("Unsupported unary operator.")
            operand = eval_node(n.operand)
            try:
                return float(unary_ops[op_type](operand))
            except Exception as e:
                raise ValueError("Invalid unary operation.") from e
        if isinstance(n, ast.Num):  # Py<3.8
            if isinstance(n.n, bool):
                raise ValueError("Booleans not allowed.")
            return float(n.n)
        if isinstance(n, ast.Constant):  # Py>=3.8
            if isinstance(n.value, bool):
                raise ValueError("Booleans not allowed.")
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise ValueError("Invalid constant in expression.")
        if isinstance(n, ast.Name):
            name = n.id
            if name not in variables:
                raise ValueError(f"Undefined variable: {name}")
            return coerce_numeric(variables[name])
        if isinstance(n, ast.Paren):  # Not actually used in Python AST; parentheses handled by structure
            return eval_node(n.value)
        # Disallow everything else
        raise ValueError("Invalid expression.")

    result = eval_node(node)
    return str(result)
