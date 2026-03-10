import ast
import logging

BLOCKED_MSG = 'Execution Blocked!'


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("user_code_executor")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler("execution.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


LOGGER = _configure_logger()


class SafeExpressionValidator(ast.NodeVisitor):
    allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
    allowed_unary_ops = (ast.UAdd, ast.USub)
    allowed_comp_ops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn)

    def __init__(self):
        super().__init__()
        self.operations = []

    @staticmethod
    def _op_name(op) -> str:
        return type(op).__name__

    def visit_Expression(self, node):
        self.visit(node.body)

    def visit_Constant(self, node):
        if not isinstance(node.value, (int, float, str, bool, type(None))):
            raise ValueError("Unsupported constant")
        self.operations.append(f"Constant:{type(node.value).__name__}")

    def visit_BinOp(self, node):
        if not isinstance(node.op, self.allowed_bin_ops):
            raise ValueError("Unsupported operator")
        self.operations.append(f"BinOp:{self._op_name(node.op)}")
        # Guard against excessive string repetition when both operands are constants
        if isinstance(node.op, ast.Mult):
            left = node.left
            right = node.right
            if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
                if isinstance(left.value, str) and isinstance(right.value, int):
                    if abs(right.value) > 10000:
                        raise ValueError("Excessive repetition")
                if isinstance(right.value, str) and isinstance(left.value, int):
                    if abs(left.value) > 10000:
                        raise ValueError("Excessive repetition")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        if not isinstance(node.op, self.allowed_unary_ops):
            raise ValueError("Unsupported unary operator")
        self.operations.append(f"UnaryOp:{self._op_name(node.op)}")
        self.visit(node.operand)

    def visit_Compare(self, node):
        for op in node.ops:
            if not isinstance(op, self.allowed_comp_ops):
                raise ValueError("Unsupported comparison")
            self.operations.append(f"Compare:{self._op_name(op)}")
        self.visit(node.left)
        for comp in node.comparators:
            self.visit(comp)

    def visit_Subscript(self, node):
        self.operations.append("Subscript")
        self.visit(node.value)
        sl = node.slice
        if isinstance(sl, ast.Slice):
            if sl.lower is not None:
                self.visit(sl.lower)
            if sl.upper is not None:
                self.visit(sl.upper)
            if sl.step is not None:
                self.visit(sl.step)
        else:
            self.visit(sl)

    def visit_List(self, node):
        raise ValueError("Lists are not allowed")

    def visit_Tuple(self, node):
        raise ValueError("Tuples are not allowed")

    def visit_Dict(self, node):
        raise ValueError("Dicts are not allowed")

    def visit_Set(self, node):
        raise ValueError("Sets are not allowed")

    def visit_Name(self, node):
        raise ValueError("Names are not allowed")

    def visit_Call(self, node):
        raise ValueError("Function calls are not allowed")

    def visit_Attribute(self, node):
        raise ValueError("Attribute access is not allowed")

    def visit_IfExp(self, node):
        raise ValueError("Conditional expressions are not allowed")

    def visit_Lambda(self, node):
        raise ValueError("Lambdas are not allowed")

    def visit_BoolOp(self, node):
        raise ValueError("Boolean operations are not allowed")

    def visit_JoinedStr(self, node):
        raise ValueError("f-strings are not allowed")

    def visit_FormattedValue(self, node):
        raise ValueError("f-strings are not allowed")

    def visit_NamedExpr(self, node):
        raise ValueError("Walrus operator is not allowed")

    def generic_visit(self, node):
        if isinstance(node, ast.Load):
            return
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def parse_user_expression(user_code: str) -> ast.AST:
    """
    Parse a user-provided string into an AST Expression.
    Raises SyntaxError or TypeError on invalid input.
    """
    if not isinstance(user_code, str):
        raise TypeError("user_code must be a string")
    return ast.parse(user_code, mode='eval')


def validate_user_ast(tree: ast.AST):
    """
    Validate the AST to ensure it only contains safe operations.
    Returns a list of operation descriptions.
    Raises ValueError on unsafe constructs.
    """
    validator = SafeExpressionValidator()
    validator.visit(tree)
    return validator.operations


def evaluate_safe_expression(tree: ast.AST):
    """
    Evaluate a previously validated AST expression in a restricted environment.
    May raise runtime exceptions (e.g., ZeroDivisionError).
    """
    code_obj = compile(tree, filename="<user_code>", mode="eval")
    return eval(code_obj, {"__builtins__": {}}, {})


def execute_user_code(user_code: str) -> str:
    LOGGER.info(f"Attempt - user_code: {user_code!r}")
    if not isinstance(user_code, str):
        LOGGER.warning("Execution Blocked - reason: non-string input")
        return BLOCKED_MSG
    try:
        tree = parse_user_expression(user_code)
    except Exception as e:
        LOGGER.warning(f"Execution Blocked at parse - reason: {type(e).__name__}: {e}")
        return BLOCKED_MSG
    try:
        operations = validate_user_ast(tree)
        if operations:
            LOGGER.info("Validated operations: " + ", ".join(operations))
        else:
            LOGGER.info("Validated operations: (none)")
    except Exception as e:
        LOGGER.warning(f"Execution Blocked at validation - reason: {type(e).__name__}: {e}")
        return BLOCKED_MSG
    try:
        result = evaluate_safe_expression(tree)
    except Exception as e:
        LOGGER.warning(f"Execution Blocked at evaluation - reason: {type(e).__name__}: {e}")
        return BLOCKED_MSG
    try:
        result_str = str(result)
    except Exception as e:
        LOGGER.warning(f"Execution Blocked at stringify - reason: {type(e).__name__}: {e}")
        return BLOCKED_MSG
    LOGGER.info(f"Success - result: {result_str!r}")
    return result_str
