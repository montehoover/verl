import re
import operator
import ast
from decimal import Decimal, DivisionByZero, InvalidOperation, getcontext
from typing import Dict, List, Union, Any

# Set a reasonable precision for decimal operations
getcontext().prec = 28

Token = Union[str, Decimal]

# Regex for validating variable names (identifiers)
_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def simplify_math_expression(formula_str: str, vars_mapping: Dict[str, Any] | None = None) -> str:
    """
    Compute and simplify a mathematical expression string with variables, returning the result as a string.

    Parameters:
    - formula_str: A string representing a mathematical formula. Supports +, -, *, /, parentheses,
                   unary +/-, numeric literals, and variables.
    - vars_mapping: A dictionary mapping variable names to numeric values (int, float, Decimal, or numeric strings).

    Returns:
    - The computed result as a string (integers without decimal point, decimals trimmed).

    Raises:
    - ValueError for any invalid input, unknown variables, unsupported operations, or arithmetic errors.
    """
    if not isinstance(formula_str, str):
        raise ValueError("formula_str must be a string")

    expr = formula_str.strip()
    if not expr:
        raise ValueError("Empty expression")

    variables: Dict[str, Any] = vars_mapping or {}

    # Allowed operator mappings for AST nodes
    BIN_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    UNARY_OPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def coerce_constant(value: Any) -> Decimal:
        if isinstance(value, bool):
            # Disallow booleans (they are ints in Python but not desired here)
            raise ValueError("Invalid constant")
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int):
            return Decimal(value)
        if isinstance(value, float):
            return Decimal(str(value))
        if isinstance(value, str):
            return Decimal(value)
        raise ValueError("Invalid constant")

    def eval_node(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in BIN_OPS:
                raise ValueError("Unsupported operator")
            left = eval_node(node.left)
            right = eval_node(node.right)
            if op_type is ast.Div and right == 0:
                raise DivisionByZero("division by zero")
            return BIN_OPS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in UNARY_OPS:
                raise ValueError("Unsupported unary operator")
            operand = eval_node(node.operand)
            return UNARY_OPS[op_type](operand)

        # Numeric literals
        if isinstance(node, ast.Num):  # Py<3.8 compatibility
            return coerce_constant(node.n)

        if isinstance(node, ast.Constant):
            return coerce_constant(node.value)

        if isinstance(node, ast.Name):
            ident = node.id
            if not _VAR_NAME_RE.match(ident):
                raise ValueError(f"Invalid variable name: {ident}")
            if ident not in variables:
                raise ValueError(f"Unknown variable: {ident}")
            return _coerce_to_decimal(variables[ident])

        # Explicitly reject all other node types
        raise ValueError("Unsupported expression")

    try:
        tree = ast.parse(expr, mode="eval")
        result = eval_node(tree)

        # Normalize output formatting: integers without decimal point, decimals trimmed
        if result == 0:
            return "0"

        if result == result.to_integral_value():
            return str(int(result))

        s = format(result, "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s if s else "0"
    except (InvalidOperation, DivisionByZero, ArithmeticError, ValueError, SyntaxError) as e:
        raise ValueError("Invalid expression") from e


def evaluate_expression(expression: str, variables: Dict[str, Any] | None = None) -> str:
    """
    Evaluate a simple mathematical expression string containing +, -, *, / and parentheses.
    Optionally substitute variables using the provided 'variables' dictionary mapping variable
    names to numeric values (int, float, Decimal, or numeric strings).
    Returns the result as a string. Raises ValueError for invalid expressions.
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string")

    expr = expression.strip()
    if not expr:
        raise ValueError("Empty expression")

    try:
        tokens = _tokenize(expr)
        rpn = _to_rpn(tokens, variables or {})
        result = _eval_rpn(rpn)

        # Normalize output formatting: integers without decimal point, decimals trimmed
        # Handle negative zero
        if result == 0:
            return "0"

        # If it's an integer value, return as integer string
        if result == result.to_integral_value():
            return str(int(result))

        # Otherwise, fixed-point string without trailing zeros
        s = format(result, "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s if s else "0"
    except (InvalidOperation, DivisionByZero, ArithmeticError, ValueError) as e:
        raise ValueError("Invalid expression") from e


def _tokenize(expr: str) -> List[str]:
    """
    Convert the expression string into a list of tokens (numbers, identifiers, operators, parentheses).
    Handles unary + and - by inserting a leading 0 (e.g., -x -> 0 - x).
    """
    tokens: List[str] = []
    i = 0
    n = len(expr)

    def is_digit_or_dot(c: str) -> bool:
        return c.isdigit() or c == "."

    def is_identifier_start(c: str) -> bool:
        return c.isalpha() or c == "_"

    def is_identifier_char(c: str) -> bool:
        return c.isalnum() or c == "_"

    prev_token: str | None = None  # Track previous non-space token

    while i < n:
        c = expr[i]

        if c.isspace():
            i += 1
            continue

        if c in "()+-*/":
            # Handle unary +/-
            if c in "+-" and (prev_token is None or prev_token in "+-*/("):
                # Insert a leading 0 to convert unary into binary
                tokens.append("0")
                tokens.append(c)
                prev_token = c
                i += 1
                continue

            tokens.append(c)
            prev_token = c
            i += 1
            continue

        # Identifier (variable)
        if is_identifier_start(c):
            start = i
            i += 1
            while i < n and is_identifier_char(expr[i]):
                i += 1
            ident = expr[start:i]
            tokens.append(ident)
            prev_token = ident
            continue

        # Number (integer or decimal)
        if is_digit_or_dot(c):
            start = i
            dot_count = 0
            while i < n and is_digit_or_dot(expr[i]):
                if expr[i] == ".":
                    dot_count += 1
                    if dot_count > 1:
                        raise ValueError("Invalid number format")
                i += 1

            num_str = expr[start:i]
            # Validate that it's not just '.' or empty
            if num_str == ".":
                raise ValueError("Invalid number format")
            tokens.append(num_str)
            prev_token = num_str
            continue

        # Any other character is invalid
        raise ValueError(f"Invalid character: {c}")

    return tokens


def _to_rpn(tokens: List[str], variables: Dict[str, Any]) -> List[Token]:
    """
    Convert list of tokens to Reverse Polish Notation using the shunting-yard algorithm.
    Identifiers are looked up in 'variables' and converted to Decimal.
    """
    output: List[Token] = []
    ops_stack: List[str] = []

    precedence = {"+": 1, "-": 1, "*": 2, "/": 2}
    left_assoc = {"+", "-", "*", "/"}

    for tok in tokens:
        if _is_number_token(tok):
            output.append(Decimal(tok))
        elif _is_identifier_token(tok):
            if tok not in variables:
                raise ValueError(f"Unknown variable: {tok}")
            output.append(_coerce_to_decimal(variables[tok]))
        elif tok in precedence:
            while (
                ops_stack
                and ops_stack[-1] in precedence
                and (
                    (ops_stack[-1] in left_assoc and precedence[ops_stack[-1]] >= precedence[tok])
                )
            ):
                output.append(ops_stack.pop())
            ops_stack.append(tok)
        elif tok == "(":
            ops_stack.append(tok)
        elif tok == ")":
            while ops_stack and ops_stack[-1] != "(":
                output.append(ops_stack.pop())
            if not ops_stack or ops_stack[-1] != "(":
                raise ValueError("Mismatched parentheses")
            ops_stack.pop()  # Discard '('
        else:
            raise ValueError(f"Invalid token: {tok}")

    while ops_stack:
        op = ops_stack.pop()
        if op in ("(", ")"):
            raise ValueError("Mismatched parentheses")
        output.append(op)

    return output


def _eval_rpn(rpn: List[Token]) -> Decimal:
    """
    Evaluate an RPN expression using Decimal arithmetic.
    """
    stack: List[Decimal] = []

    for tok in rpn:
        if isinstance(tok, Decimal):
            stack.append(tok)
        else:
            # Operator
            if len(stack) < 2:
                raise ValueError("Insufficient operands")
            b = stack.pop()
            a = stack.pop()
            if tok == "+":
                stack.append(a + b)
            elif tok == "-":
                stack.append(a - b)
            elif tok == "*":
                stack.append(a * b)
            elif tok == "/":
                if b == 0:
                    raise DivisionByZero("division by zero")
                stack.append(a / b)
            else:
                raise ValueError(f"Unknown operator: {tok}")

    if len(stack) != 1:
        raise ValueError("Invalid expression")
    return stack[0]


def _is_number_token(tok: str) -> bool:
    # A number token is composed of digits and at most one dot, but we already validated on tokenize
    # We still do a quick check here.
    if not tok:
        return False
    if tok.count(".") > 1:
        return False
    return all(ch.isdigit() or ch == "." for ch in tok)


def _is_identifier_token(tok: str) -> bool:
    if not tok:
        return False
    if tok[0].isalpha() or tok[0] == "_":
        return all(ch.isalnum() or ch == "_" for ch in tok[1:])
    return False


def _coerce_to_decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        # Use str to avoid binary floating errors
        return Decimal(str(value))
    if isinstance(value, str):
        # Allow numeric strings (including scientific notation)
        return Decimal(value)
    raise ValueError(f"Unsupported variable type: {type(value).__name__}")
