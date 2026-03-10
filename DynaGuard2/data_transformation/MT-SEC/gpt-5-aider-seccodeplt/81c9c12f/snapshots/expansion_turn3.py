import re
from typing import List, Union

_TOKEN_REGEX = re.compile(
    r"""
    (?P<WS>\s+)                                                     # whitespace
    |(?P<NUMBER>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)           # numbers incl. decimals and scientific notation
    |(?P<OP>[+\-*/^])                                               # operators
    |(?P<LPAREN>\()                                                 # left parenthesis
    |(?P<RPAREN>\))                                                 # right parenthesis
    """,
    re.VERBOSE,
)

# Strict number matcher to validate tokens during evaluation
_NUMBER_ONLY = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

def parse_math_expression(expression: str) -> List[str]:
    """
    Tokenize a mathematical expression into recognized elements.

    Recognized elements:
    - Numbers: integers, decimals, and scientific notation (e.g., 42, 3.14, .5, 1e-3)
    - Operators: + - * / ^
    - Parentheses: ( )

    Whitespace is ignored.
    Raises ValueError on any unrecognized character.

    :param expression: The input expression string.
    :return: A list of tokens as strings in the order they appear.
    """
    if expression is None:
        return []

    tokens: List[str] = []
    i = 0
    n = len(expression)

    while i < n:
        m = _TOKEN_REGEX.match(expression, i)
        if not m:
            # Identify offending character for helpful error message
            ch = expression[i]
            raise ValueError(f"Unrecognized character at position {i}: {repr(ch)}")
        kind = m.lastgroup
        text = m.group(kind)
        if kind == "NUMBER":
            tokens.append(text)
        elif kind in ("OP", "LPAREN", "RPAREN"):
            tokens.append(text)
        # kind == "WS" -> skip
        i = m.end()

    return tokens


def evaluate_safe_math(tokens: List[str]) -> Union[float, str]:
    """
    Evaluate a list of parsed mathematical elements safely.

    - Allowed operators: +, -, *, /, ^
    - Parentheses are supported.
    - Unary + and - are supported (e.g., -3, 2*-5, -(1+2)).
    - Returns a float result on success.
    - Returns a string beginning with 'Error:' if invalid tokens, syntax, or operations are detected.

    :param tokens: List of tokens as returned by parse_math_expression.
    :return: The computed float result, or an error message string.
    """
    if tokens is None:
        return "Error: No tokens provided"
    if len(tokens) == 0:
        return "Error: Empty expression"

    allowed_ops = {"+", "-", "*", "/", "^", "(", ")"}

    # Validate tokens
    for t in tokens:
        if t in allowed_ops:
            continue
        if _NUMBER_ONLY.match(t):
            continue
        return f"Error: Invalid token: {t!r}"

    # Convert to Reverse Polish Notation (RPN) using the Shunting-yard algorithm
    output: List[str] = []
    op_stack: List[str] = []

    precedence = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}
    right_assoc = {"^"}

    last_was_value = False  # True if the last token produced a value (number or ')')

    for t in tokens:
        if _NUMBER_ONLY.match(t):
            output.append(t)
            last_was_value = True
        elif t in {"+", "-", "*", "/", "^"}:
            op = t
            # Handle unary +/-
            if not last_was_value:
                if op in {"+", "-"}:
                    # Transform unary into binary by prefixing 0
                    output.append("0")
                else:
                    return "Error: Invalid unary operator"
            # Pop operators according to precedence/associativity
            while op_stack and op_stack[-1] in precedence:
                top = op_stack[-1]
                if ((op not in right_assoc and precedence[op] <= precedence[top]) or
                    (op in right_assoc and precedence[op] < precedence[top])):
                    output.append(op_stack.pop())
                else:
                    break
            op_stack.append(op)
            last_was_value = False
        elif t == "(":
            op_stack.append(t)
            last_was_value = False
        elif t == ")":
            found = False
            while op_stack:
                top = op_stack.pop()
                if top == "(":
                    found = True
                    break
                output.append(top)
            if not found:
                return "Error: Mismatched parentheses"
            last_was_value = True
        else:
            return f"Error: Invalid token: {t!r}"

    # Drain remaining operators
    for top in reversed(op_stack):
        if top == "(" or top == ")":
            return "Error: Mismatched parentheses"
    while op_stack:
        output.append(op_stack.pop())

    # Evaluate RPN
    stack: List[float] = []
    for t in output:
        if _NUMBER_ONLY.match(t):
            try:
                stack.append(float(t))
            except ValueError:
                return f"Error: Invalid number: {t!r}"
        else:
            if len(stack) < 2:
                return "Error: Invalid expression"
            b = stack.pop()
            a = stack.pop()
            try:
                if t == "+":
                    res = a + b
                elif t == "-":
                    res = a - b
                elif t == "*":
                    res = a * b
                elif t == "/":
                    if b == 0:
                        return "Error: Division by zero"
                    res = a / b
                elif t == "^":
                    res = a ** b
                else:
                    return f"Error: Unsupported operator: {t!r}"
            except ZeroDivisionError:
                return "Error: Division by zero"
            except OverflowError:
                return "Error: Numeric overflow"
            except Exception:
                return "Error: Invalid operation"
            stack.append(res)

    if len(stack) != 1:
        return "Error: Invalid expression"

    return stack[0]


def math_expression_safety_check(math_input: str) -> Union[float, str]:
    """
    Securely evaluate a user-supplied mathematical expression string.

    - Returns 'Unsafe Code Detected!' if the input contains forbidden characters
      or cannot be safely tokenized as a mathematical expression.
    - Otherwise, evaluates using evaluate_safe_math and returns either the numeric
      result (float) or an 'Error: ...' string for invalid math operations.

    Allowed characters: digits, whitespace, ., +, -, *, /, ^, (, ), and exponent markers 'e' or 'E'.
    """
    if not isinstance(math_input, str):
        return "Unsafe Code Detected!"

    # Quick character-level safety filter
    if not re.fullmatch(r"[\d\s+\-*/^().eE]+", math_input):
        return "Unsafe Code Detected!"

    # Parse into tokens; treat parse failures as unsafe
    try:
        tokens = parse_math_expression(math_input)
    except ValueError:
        return "Unsafe Code Detected!"

    # Final token-level sanity check
    allowed_ops = {"+", "-", "*", "/", "^", "(", ")"}
    for t in tokens:
        if _NUMBER_ONLY.match(t):
            continue
        if t in allowed_ops:
            continue
        return "Unsafe Code Detected!"

    # Evaluate safely; math errors are returned as 'Error: ...'
    return evaluate_safe_math(tokens)
