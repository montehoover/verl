import ast

BLOCKED_MSG = 'Execution Blocked!'


class SafeExpressionValidator(ast.NodeVisitor):
    allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
    allowed_unary_ops = (ast.UAdd, ast.USub)
    allowed_comp_ops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn)

    def visit_Expression(self, node):
        self.visit(node.body)

    def visit_Constant(self, node):
        if not isinstance(node.value, (int, float, str, bool, type(None))):
            raise ValueError("Unsupported constant")

    def visit_BinOp(self, node):
        if not isinstance(node.op, self.allowed_bin_ops):
            raise ValueError("Unsupported operator")
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
        self.visit(node.operand)

    def visit_Compare(self, node):
        for op in node.ops:
            if not isinstance(op, self.allowed_comp_ops):
                raise ValueError("Unsupported comparison")
        self.visit(node.left)
        for comp in node.comparators:
            self.visit(comp)

    def visit_Subscript(self, node):
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


def execute_user_code(user_code: str) -> str:
    if not isinstance(user_code, str):
        return BLOCKED_MSG
    try:
        tree = ast.parse(user_code, mode='eval')
    except Exception:
        return BLOCKED_MSG
    try:
        SafeExpressionValidator().visit(tree)
    except Exception:
        return BLOCKED_MSG
    try:
        result = eval(compile(tree, filename="<user_code>", mode="eval"), {"__builtins__": {}}, {})
    except Exception:
        return BLOCKED_MSG
    try:
        return str(result)
    except Exception:
        return BLOCKED_MSG
