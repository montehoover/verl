import re
import math
import ast
from typing import List, Tuple, Union

Token = Tuple[str, str]  # (type, value)
UNSAFE_MSG = "Unsafe Operation Detected!"


def _is_safe_expression(expression: str) -> bool:
    if not isinstance(expression, str):
        return False
    if expression is None or len(expression) == 0:
        return False

    # Only allow digits, whitespace, decimal point, basic ops, and parentheses
    allowed_extra = set(".+-*/()")
    for ch in expression:
        if ch.isdigit() or ch.isspace() or ch in allowed_extra:
            continue
        return False

    # Disallow potentially unsafe operator sequences
    if "**" in expression or "//" in expression:
        return False

    # Parentheses must be balanced and never go negative
    balance = 0
    for ch in expression:
        if ch == '(':
            balance += 1
        elif ch == ')':
            balance -= 1
            if balance < 0:
                return False
    if balance != 0:
        return False

    return True


def _tokenize(expression: str) -> List[Token]:
    if expression is None:
        raise ValueError("Expression cannot be None")

    token_specification = [
        ("NUMBER", r'(?:\d+(?:\.\d*)?|\.\d+)'),
        ("OP", r'[+\-*/]'),
        ("LPAREN", r'\('),
        ("RPAREN", r'\)'),
        ("SKIP", r'\s+'),
        ("MISMATCH", r'.'),
    ]
    tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)
    tokens: List[Token] = []
    for mo in re.finditer(tok_regex, expression):
        kind = mo.lastgroup
        value = mo.group()
        if kind == "SKIP":
            continue
        elif kind == "MISMATCH":
            raise ValueError(f"Unexpected character '{value}' at position {mo.start()}")
        else:
            tokens.append((kind, value))
    tokens.append(("EOF", ""))  # End marker
    return tokens


class _Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> None:
        if self.pos < len(self.tokens) - 1:
            self.pos += 1

    def expect(self, kind: str) -> Token:
        tok = self.current()
        if tok[0] != kind:
            raise ValueError(f"Expected {kind}, got {tok[0]}")
        self.advance()
        return tok

    def parse(self) -> float:
        if self.current()[0] == "EOF":
            raise ValueError("Empty expression")
        value = self.parse_expression()
        if self.current()[0] != "EOF":
            raise ValueError("Unexpected input after complete expression")
        return value

    # expression -> term ((+|-) term)*
    def parse_expression(self) -> float:
        value = self.parse_term()
        while self.current()[0] == "OP" and self.current()[1] in ('+', '-'):
            op = self.current()[1]
            self.advance()
            rhs = self.parse_term()
            if op == '+':
                value += rhs
            else:
                value -= rhs
        return value

    # term -> factor ((*|/) factor)*
    def parse_term(self) -> float:
        value = self.parse_factor()
        while self.current()[0] == "OP" and self.current()[1] in ('*', '/'):
            op = self.current()[1]
            self.advance()
            rhs = self.parse_factor()
            if op == '*':
                value *= rhs
            else:
                if rhs == 0:
                    raise ZeroDivisionError("division by zero")
                value /= rhs
        return value

    # factor -> NUMBER | '(' expression ')' | (+|-) factor
    def parse_factor(self) -> float:
        tok_type, tok_val = self.current()

        # Unary plus/minus
        if tok_type == "OP" and tok_val in ('+', '-'):
            op = tok_val
            self.advance()
            val = self.parse_factor()
            return val if op == '+' else -val

        if tok_type == "NUMBER":
            self.advance()
            return float(tok_val)

        if tok_type == "LPAREN":
            self.advance()
            val = self.parse_expression()
            self.expect("RPAREN")
            return val

        raise ValueError(f"Unexpected token: {tok_type} '{tok_val}'")


def calculate_expression(expression: str):
    """
    Evaluate a simple arithmetic expression string.

    Supports:
    - Addition (+), subtraction (-), multiplication (*), division (/)
    - Operator precedence and parentheses
    - Unary plus/minus (e.g., -3, +4)

    Returns:
    - float result on success
    - 'Unsafe Operation Detected!' on invalid or unsafe input/operation
    """
    if not _is_safe_expression(expression):
        return UNSAFE_MSG

    try:
        tokens = _tokenize(expression)
        parser = _Parser(tokens)
        result = parser.parse()
        if not math.isfinite(result):
            return UNSAFE_MSG
        return result
    except (ValueError, ZeroDivisionError, OverflowError):
        return UNSAFE_MSG


# --- Safe Python expression executor utility ---

_SAFE_EXPR_MAX_LEN = 2000
_SAFE_AST_MAX_NODES = 300

_SAFE_BUILTINS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
}

_SAFE_MATH_NAMES = [
    "pi", "e", "tau", "inf", "nan",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sqrt",
    "log", "log10", "log2",
    "exp",
    "floor", "ceil", "trunc",
    "degrees", "radians",
    "hypot",
    "copysign",
    "fabs",
    "fmod", "remainder",
]

_SAFE_ENV: dict = {}
_SAFE_ENV.update(_SAFE_BUILTINS)
for _name in _SAFE_MATH_NAMES:
    if hasattr(math, _name):
        _SAFE_ENV[_name] = getattr(math, _name)
_SAFE_ALLOWED_NAMES = set(_SAFE_ENV.keys())


class _SafeExpressionValidator(ast.NodeVisitor):
    def __init__(self, allowed_names: set, max_nodes: int):
        self.allowed_names = allowed_names
        self.max_nodes = max_nodes
        self._count = 0

    def _tick(self, node: ast.AST):
        self._count += 1
        if self._count > self.max_nodes:
            raise ValueError("Expression too complex")

    def generic_visit(self, node: ast.AST):
        self._tick(node)
        super().generic_visit(node)

    # Allowed constructs
    def visit_Expression(self, node: ast.Expression):
        self._tick(node)
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp):
        self._tick(node)
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
            raise ValueError("Operator not allowed")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self._tick(node)
        if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            raise ValueError("Unary operator not allowed")
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp):
        self._tick(node)
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ValueError("Boolean operator not allowed")
        for v in node.values:
            self.visit(v)

    def visit_Compare(self, node: ast.Compare):
        self._tick(node)
        allowed = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
        for op in node.ops:
            if not isinstance(op, allowed):
                raise ValueError("Comparison operator not allowed")
        self.visit(node.left)
        for c in node.comparators:
            self.visit(c)

    def visit_IfExp(self, node: ast.IfExp):
        self._tick(node)
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Name(self, node: ast.Name):
        self._tick(node)
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("Assignments not allowed")
        if node.id not in self.allowed_names:
            raise ValueError("Name not allowed")

    def visit_Call(self, node: ast.Call):
        self._tick(node)
        # Only calls to whitelisted simple names
        if not isinstance(node.func, ast.Name) or node.func.id not in self.allowed_names:
            raise ValueError("Function call not allowed")
        # No starargs or kwargs
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            raise ValueError("Starred arguments not allowed")
        if node.keywords:
            raise ValueError("Keyword arguments not allowed")
        for arg in node.args:
            self.visit(arg)

    def visit_Constant(self, node: ast.Constant):
        self._tick(node)
        if not isinstance(node.value, (int, float, bool)):
            raise ValueError("Constant type not allowed")

    # Disallowed constructs
    def visit_Attribute(self, node: ast.Attribute):
        raise ValueError("Attribute access not allowed")

    def visit_Subscript(self, node: ast.Subscript):
        raise ValueError("Subscript not allowed")

    def visit_List(self, node: ast.List):
        raise ValueError("List literals not allowed")

    def visit_Tuple(self, node: ast.Tuple):
        raise ValueError("Tuple literals not allowed")

    def visit_Dict(self, node: ast.Dict):
        raise ValueError("Dict literals not allowed")

    def visit_Set(self, node: ast.Set):
        raise ValueError("Set literals not allowed")

    def visit_ListComp(self, node: ast.ListComp):
        raise ValueError("Comprehensions not allowed")

    def visit_SetComp(self, node: ast.SetComp):
        raise ValueError("Comprehensions not allowed")

    def visit_DictComp(self, node: ast.DictComp):
        raise ValueError("Comprehensions not allowed")

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        raise ValueError("Generators not allowed")

    def visit_Lambda(self, node: ast.Lambda):
        raise ValueError("Lambdas not allowed")

    def visit_NamedExpr(self, node: ast.NamedExpr):
        raise ValueError("Assignment expressions not allowed")

    def visit_Await(self, node: ast.Await):
        raise ValueError("Await not allowed")

    def visit_Yield(self, node: ast.Yield):
        raise ValueError("Yield not allowed")

    def visit_YieldFrom(self, node: ast.YieldFrom):
        raise ValueError("Yield not allowed")


def safely_execute_expression(expression_code: str):
    """
    Safely execute a user-supplied Python expression string.

    Returns:
    - The evaluated result on success
    - 'Unsafe Operation Detected!' on unsafe input or errors
    """
    try:
        if not isinstance(expression_code, str):
            return UNSAFE_MSG
        if expression_code is None:
            return UNSAFE_MSG
        # Basic size/length limits
        if len(expression_code) == 0 or len(expression_code) > _SAFE_EXPR_MAX_LEN:
            return UNSAFE_MSG

        # Parse as an expression
        try:
            tree = ast.parse(expression_code, mode="eval")
        except SyntaxError:
            return UNSAFE_MSG

        # Validate AST
        validator = _SafeExpressionValidator(_SAFE_ALLOWED_NAMES, _SAFE_AST_MAX_NODES)
        validator.visit(tree)

        # Evaluate in restricted environment
        result = eval(compile(tree, "<safe-expr>", "eval"), {"__builtins__": {}}, dict(_SAFE_ENV))

        # Post-check result
        if isinstance(result, float):
            if not math.isfinite(result):
                return UNSAFE_MSG
        elif isinstance(result, int):
            # Avoid monstrously large integers (resource exhaustion)
            if len(str(abs(result))) > 10000:
                return UNSAFE_MSG
        elif isinstance(result, bool):
            pass
        else:
            # Only numeric/boolean results are allowed
            return UNSAFE_MSG

        return result
    except Exception:
        return UNSAFE_MSG
