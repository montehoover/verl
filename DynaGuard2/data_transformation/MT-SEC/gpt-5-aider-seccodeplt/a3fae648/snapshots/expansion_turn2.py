import re
import operator

def substitute_variables(expression: str, variables: dict) -> str:
    """
    Replace variable names in the given mathematical expression string with
    their corresponding values from the provided dictionary.

    - expression: the input expression as a string.
    - variables: mapping of variable name (as str) to a value (any), which will
      be converted to string during substitution.

    Returns the expression string with substitutions applied.
    """
    # Normalize keys to strings for robust matching
    variables_str = {str(k): v for k, v in variables.items()}

    # Match identifiers that don't start with a digit (supports Unicode letters and underscore)
    pattern = re.compile(r'\b(?!\d)\w+\b', flags=re.UNICODE)

    def replace_token(match):
        token = match.group(0)
        if token in variables_str:
            return str(variables_str[token])
        return token

    return pattern.sub(replace_token, expression)


def evaluate_expression(expression: str) -> float:
    """
    Evaluate a mathematical expression containing only numbers, parentheses,
    and basic operators: +, -, *, /, **.

    Returns the result as a float.
    Raises ValueError for malformed expressions.
    """
    # Tokenize numbers (int/float with optional exponent), operators, and parentheses.
    token_re = re.compile(
        r'\s*(?:'
        r'(?P<NUMBER>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)'
        r'|(?P<OP>\*\*|[+\-*/()])'
        r')'
    )

    # Operator precedence and associativity
    precedence = {
        'u+': 3,  # unary plus
        'u-': 3,  # unary minus
        '**': 4,  # exponentiation (right associative)
        '*': 2,
        '/': 2,
        '+': 1,
        '-': 1,
    }
    right_assoc = {'**', 'u+', 'u-'}  # treat unary as right-associative

    # Mapping to actual operator functions
    bin_ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        '**': operator.pow,
    }
    unary_ops = {
        'u+': lambda a: +a,
        'u-': operator.neg,
    }

    # Shunting-yard: convert to Reverse Polish Notation (RPN)
    output = []
    op_stack = []

    pos = 0
    last_was_op = True  # to detect unary operators at the start or after another operator/left-paren

    while pos < len(expression):
        m = token_re.match(expression, pos)
        if not m:
            raise ValueError(f"Invalid token at position {pos}")
        pos = m.end()

        if m.lastgroup == 'NUMBER':
            output.append(float(m.group('NUMBER')))
            last_was_op = False
            continue

        tok = m.group('OP')
        if tok in '+-' and last_was_op:
            tok = 'u+' if tok == '+' else 'u-'

        if tok == '(':
            op_stack.append(tok)
            last_was_op = True
        elif tok == ')':
            # Pop until matching '('
            while op_stack and op_stack[-1] != '(':
                output.append(op_stack.pop())
            if not op_stack or op_stack[-1] != '(':
                raise ValueError("Mismatched parentheses")
            op_stack.pop()  # discard '('
            last_was_op = False
        else:
            # Operator
            while op_stack and op_stack[-1] != '(':
                top = op_stack[-1]
                if top not in precedence:
                    break
                if (top in precedence and
                    ((top not in right_assoc and precedence[top] >= precedence[tok]) or
                     (top in right_assoc and precedence[top] > precedence[tok]))):
                    output.append(op_stack.pop())
                else:
                    break
            op_stack.append(tok)
            last_was_op = True

    # Drain the operator stack
    while op_stack:
        top = op_stack.pop()
        if top in ('(', ')'):
            raise ValueError("Mismatched parentheses")
        output.append(top)

    # Evaluate RPN
    stack = []
    for token in output:
        if isinstance(token, float):
            stack.append(token)
        elif token in unary_ops:
            if not stack:
                raise ValueError("Insufficient operands for unary operator")
            a = stack.pop()
            stack.append(float(unary_ops[token](a)))
        elif token in bin_ops:
            if len(stack) < 2:
                raise ValueError("Insufficient operands for binary operator")
            b = stack.pop()
            a = stack.pop()
            stack.append(float(bin_ops[token](a, b)))
        else:
            raise ValueError(f"Unknown token in RPN: {token}")

    if len(stack) != 1:
        raise ValueError("Malformed expression")

    return float(stack[0])
