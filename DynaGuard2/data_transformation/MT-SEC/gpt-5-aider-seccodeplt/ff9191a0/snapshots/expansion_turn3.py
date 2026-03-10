import re
import operator
import ast
from typing import List, Dict, Union

# Precompiled regular expression for tokenizing mathematical expressions.
# Tokens include:
# - NUMBER: integers, decimals, and scientific notation (e.g., 3, 3.14, .5, 1e-3)
# - IDENTIFIER: variable/function names (e.g., x, alpha1, _temp)
# - OP: operators and punctuation (e.g., +, -, *, /, ^, %, (, ), ,, =, <, >, <=, >=, ==, !=, &&, ||)
# - WS: whitespace (ignored)
_TOKEN_REGEX = re.compile(
    r"""
    (?P<NUMBER>
        (?:
            (?:\d+\.\d*|\d*\.\d+|\d+)
            (?:[eE][+\-]?\d+)?
        )
    )
    | (?P<IDENTIFIER>[A-Za-z_][A-Za-z_0-9]*)
    | (?P<OP>
        (?:<=|>=|==|!=|&&|\|\|)
        | [+\-*/^%=(),<>]
      )
    | (?P<WS>\s+)
    | (?P<MISMATCH>.)
    """,
    re.VERBOSE,
)

# Identifier pattern used to detect variable tokens (not numbers or operators)
_IDENTIFIER_RE = re.compile(r'^[A-Za-z_][A-Za-z_0-9]*$')

# Numeric literal pattern for coercing string variable values like "-3.14" or "1e-5"
_NUMERIC_LITERAL_RE = re.compile(r'^[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?$')


def parse_expression(expr: str) -> List[str]:
    """
    Tokenize a mathematical expression string into a list of tokens.

    The returned tokens include:
    - Numbers (as strings), supporting integers, decimals, and scientific notation.
    - Identifiers (variable or function names).
    - Operators and punctuation: + - * / ^ % ( ) , = < > <= >= == != && ||

    Whitespace is ignored. Raises ValueError on unexpected characters.
    """
    tokens: List[str] = []
    for match in _TOKEN_REGEX.finditer(expr):
        kind = match.lastgroup
        value = match.group()
        if kind == "WS":
            continue
        elif kind in ("NUMBER", "IDENTIFIER", "OP"):
            tokens.append(value)
        elif kind == "MISMATCH":
            pos = match.start()
            raise ValueError(f"Unexpected character {value!r} at position {pos}")
    return tokens


def substitute_variables(tokens: List[str], values: Dict[str, Union[int, float, str]]) -> List[str]:
    """
    Substitute identifier tokens with their corresponding values.

    - Only tokens that are identifiers and are present as keys in 'values' are replaced.
    - Numeric values (int/float) are converted to tokens using parse_expression(str(value)),
      ensuring signs are handled consistently (e.g., -5 -> ['-', '5']).
    - String values are tokenized with parse_expression, allowing substitution with either
      a single token (e.g., '3.14') or multiple tokens (e.g., '2*x').

    Returns a new list of tokens with substitutions applied.
    """
    substituted: List[str] = []
    for tok in tokens:
        if _IDENTIFIER_RE.match(tok) and tok in values:
            val = values[tok]
            if isinstance(val, (int, float)):
                repl_tokens = parse_expression(str(val))
            elif isinstance(val, str):
                repl_tokens = parse_expression(val)
            else:
                raise TypeError(f"Unsupported value type for variable '{tok}': {type(val).__name__}")
            substituted.extend(repl_tokens)
        else:
            substituted.append(tok)
    return substituted


def _preprocess_expression(expr: str) -> str:
    """
    Preprocess the input to align with Python AST parsing:
    - Replace '^' with '**' for exponentiation.
    - Replace '&&' with 'and', '||' with 'or'.
    - Ensure there is no single '=' (assignment) present.
    """
    # Disallow single '=' (assignment). Allow '==' and '!=', '<=', '>='.
    if re.search(r'(^|[^=<>!])=(?!=)', expr):
        raise ValueError("Assignment '=' is not allowed in expressions.")
    # Logical operators
    expr = expr.replace("&&", " and ").replace("||", " or ")
    # Exponent operator
    expr = expr.replace("^", "**")
    return expr


def _coerce_variable_value(value: Union[int, float, bool, str]) -> Union[int, float, bool]:
    """
    Coerce variable values to allowed runtime types.
    Allows int, float, bool, or numeric-literal strings.
    """
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if _NUMERIC_LITERAL_RE.match(s):
            # Prefer int when possible to preserve integer semantics
            try:
                return int(s, 10)
            except ValueError:
                return float(s)
        raise ValueError(f"Variable value string is not a numeric literal: {value!r}")
    raise ValueError(f"Unsupported variable value type: {type(value).__name__}")


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _safe_eval(node: ast.AST, variables: Dict[str, Union[int, float, bool, str]]) -> Union[int, float, bool]:
    """
    Recursively and safely evaluate an AST node with a restricted set of operations.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, variables)

    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, (int, float, bool)):
            return val
        raise ValueError(f"Unsupported constant type: {type(val).__name__}")

    # For compatibility with older Python versions where numbers may be ast.Num
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        val = node.n  # type: ignore[attr-defined]
        if isinstance(val, (int, float)):
            return val
        raise ValueError(f"Unsupported numeric literal type: {type(val).__name__}")

    if isinstance(node, ast.Name):
        name = node.id
        if name not in variables:
            raise ValueError(f"Undefined variable: {name}")
        return _coerce_variable_value(variables[name])

    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        operand = _safe_eval(node.operand, variables)
        return _UNARY_OPS[type(node.op)](operand)

    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN_OPS:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = _safe_eval(node.left, variables)
        right = _safe_eval(node.right, variables)
        return _BIN_OPS[type(node.op)](left, right)

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for v in node.values:
                result = bool(_safe_eval(v, variables))
                if not result:  # short-circuit
                    return False
            return True
        elif isinstance(node.op, ast.Or):
            result = False
            for v in node.values:
                result = bool(_safe_eval(v, variables))
                if result:  # short-circuit
                    return True
            return False
        else:
            raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        left = _safe_eval(node.left, variables)
        for op, comparator in zip(node.ops, node.comparators):
            if type(op) not in _CMP_OPS:
                raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")
            right = _safe_eval(comparator, variables)
            if not _CMP_OPS[type(op)](left, right):
                return False
            left = right
        return True

    # Disallow function calls, attributes, subscripts, comprehensions, lambdas, etc.
    disallowed = (
        ast.Call, ast.Attribute, ast.Subscript, ast.List, ast.Tuple, ast.Dict, ast.Set,
        ast.ListComp, ast.SetComp, ast.DictComp if hasattr(ast, "DictComp") else tuple(),
        ast.GeneratorExp, ast.Lambda, ast.IfExp, ast.Await, ast.Yield, ast.YieldFrom,
        ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr,
        ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
        ast.With, ast.For, ast.While, ast.If, ast.Try, ast.Raise, ast.Assert,
        ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Module,
        ast.BitAnd, ast.BitOr, ast.BitXor, ast.MatMult, ast.FloorDiv,
    )
    if isinstance(node, disallowed):
        raise ValueError(f"Disallowed expression element: {type(node).__name__}")

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def evaluate_expression_safely(expression: str, variables: Dict[str, Union[int, float, bool, str]]) -> str:
    """
    Evaluate and simplify a mathematical expression securely.

    - Supports arithmetic: +, -, *, /, %, ^ (as exponent), parentheses.
    - Supports comparisons: ==, !=, <, <=, >, >=
    - Supports boolean logic: &&, || (as 'and', 'or'), and 'not'.
    - Variables are taken from the provided mapping (int, float, bool, or numeric-literal strings).
    - Raises ValueError for invalid expressions or computation errors.

    Returns:
        The computed result converted to a string.
    """
    try:
        preprocessed = _preprocess_expression(expression)
        tree = ast.parse(preprocessed, mode="eval")
        result = _safe_eval(tree, variables)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        raise ValueError(f"Invalid expression or computation error: {e}") from None
    return str(result)
