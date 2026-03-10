import ast
from typing import Union


def evaluate_arith_expression(input_expr: str) -> str:
    """
    Safely evaluate a basic arithmetic expression provided as a string.

    Allowed:
    - Numbers (int, float)
    - Binary operations: +, -, *, /, //, %, **
    - Unary operations: +, -
    - Parentheses (implicitly handled by AST structure)

    Returns:
        str: The result of the computation as a string, or 'Computation Error!' on failure.
    """
    if not isinstance(input_expr, str):
        return 'Computation Error!'

    expr = input_expr.strip()
    if not expr:
        return 'Computation Error!'

    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return 'Computation Error!'

    class _SafeEvaluator(ast.NodeVisitor):
        def visit_Expression(self, node: ast.Expression) -> Union[int, float]:
            return self.visit(node.body)

        def visit_BinOp(self, node: ast.BinOp) -> Union[int, float]:
            left = self.visit(node.left)
            right = self.visit(node.right)

            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError("Non-numeric operand")

            op = node.op
            if isinstance(op, ast.Add):
                return left + right
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
            if isinstance(op, ast.Pow):
                # Basic guard against excessively large exponentiation
                if isinstance(right, (int, float)) and abs(right) > 10000:
                    raise ValueError("Exponent too large")
                return left ** right

            raise ValueError("Unsupported binary operation")

        def visit_UnaryOp(self, node: ast.UnaryOp) -> Union[int, float]:
            operand = self.visit(node.operand)
            if not isinstance(operand, (int, float)):
                raise ValueError("Non-numeric operand")

            op = node.op
            if isinstance(op, ast.UAdd):
                return +operand
            if isinstance(op, ast.USub):
                return -operand

            raise ValueError("Unsupported unary operation")

        def visit_Constant(self, node: ast.Constant) -> Union[int, float]:
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported constant")

        # For Python versions where numbers may appear as ast.Num
        def visit_Num(self, node: ast.Num) -> Union[int, float]:  # type: ignore[override]
            if isinstance(node.n, (int, float)):
                return node.n
            raise ValueError("Unsupported number")

        # Disallow all names, calls, attributes, subscripts, etc.
        def visit_Name(self, node: ast.Name):
            raise ValueError("Names are not allowed")

        def visit_Call(self, node: ast.Call):
            raise ValueError("Function calls are not allowed")

        def visit_Attribute(self, node: ast.Attribute):
            raise ValueError("Attributes are not allowed")

        def visit_Subscript(self, node: ast.Subscript):
            raise ValueError("Subscripts are not allowed")

        def visit_List(self, node: ast.List):
            raise ValueError("Lists are not allowed")

        def visit_Tuple(self, node: ast.Tuple):
            raise ValueError("Tuples are not allowed")

        def generic_visit(self, node: ast.AST):
            # Any node type not explicitly allowed is considered malicious/unsupported
            raise ValueError(f"Unsupported node type: {type(node).__name__}")

    try:
        result = _SafeEvaluator().visit(node)
        # Convert result to string
        return str(result)
    except Exception:
        return 'Computation Error!'
