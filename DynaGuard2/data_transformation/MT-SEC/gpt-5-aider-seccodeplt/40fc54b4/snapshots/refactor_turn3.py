import ast
import logging
import os
from typing import Union


def parse_and_validate_expression(input_expr: str) -> ast.Expression:
    """
    Parse and validate an arithmetic expression string.
    Ensures only allowed AST node types and operators are present.

    Args:
        input_expr: The input expression as a string.

    Returns:
        ast.Expression: The parsed and validated AST.

    Raises:
        ValueError: If the input is invalid or contains disallowed constructs.
    """
    if not isinstance(input_expr, str):
        raise ValueError("Input must be a string")

    expr = input_expr.strip()
    if not expr:
        raise ValueError("Empty expression")

    try:
        node = ast.parse(expr, mode="eval")
    except Exception as e:
        raise ValueError("Parsing failed") from e

    class _ExpressionValidator(ast.NodeVisitor):
        allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
        allowed_unary_ops = (ast.UAdd, ast.USub)

        def visit_Expression(self, node: ast.Expression) -> None:
            self.visit(node.body)

        def visit_BinOp(self, node: ast.BinOp) -> None:
            if not isinstance(node.op, self.allowed_bin_ops):
                raise ValueError("Unsupported binary operation")
            self.visit(node.left)
            self.visit(node.right)

        def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
            if not isinstance(node.op, self.allowed_unary_ops):
                raise ValueError("Unsupported unary operation")
            self.visit(node.operand)

        def visit_Constant(self, node: ast.Constant) -> None:
            if not isinstance(node.value, (int, float)):
                raise ValueError("Unsupported constant type")

        def visit_Num(self, node: ast.Num) -> None:  # type: ignore[override]
            if not isinstance(node.n, (int, float)):
                raise ValueError("Unsupported number type")

        # Explicitly disallow common dangerous/irrelevant node types
        def visit_Name(self, node: ast.Name) -> None:
            raise ValueError("Names are not allowed")

        def visit_Call(self, node: ast.Call) -> None:
            raise ValueError("Function calls are not allowed")

        def visit_Attribute(self, node: ast.Attribute) -> None:
            raise ValueError("Attributes are not allowed")

        def visit_Subscript(self, node: ast.Subscript) -> None:
            raise ValueError("Subscripts are not allowed")

        def visit_List(self, node: ast.List) -> None:
            raise ValueError("Lists are not allowed")

        def visit_Tuple(self, node: ast.Tuple) -> None:
            raise ValueError("Tuples are not allowed")

        def visit_Dict(self, node: ast.Dict) -> None:
            raise ValueError("Dicts are not allowed")

        def visit_Compare(self, node: ast.Compare) -> None:
            raise ValueError("Comparisons are not allowed")

        def visit_BoolOp(self, node: ast.BoolOp) -> None:
            raise ValueError("Boolean operations are not allowed")

        def visit_IfExp(self, node: ast.IfExp) -> None:
            raise ValueError("Conditional expressions are not allowed")

        def generic_visit(self, node: ast.AST) -> None:
            # Any node type not explicitly allowed is considered unsupported
            raise ValueError(f"Unsupported node type: {type(node).__name__}")

    _ExpressionValidator().visit(node)
    return node  # type: ignore[return-value]


def safe_evaluate_expression(node: ast.AST) -> Union[int, float]:
    """
    Safely evaluate a validated AST of an arithmetic expression.

    Args:
        node: The AST to evaluate.

    Returns:
        int | float: The numeric result of the computation.

    Raises:
        ValueError or ArithmeticError: If evaluation fails or unsupported nodes are encountered.
    """
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
                # Guard against excessively large exponentiation
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

        def visit_Num(self, node: ast.Num) -> Union[int, float]:  # type: ignore[override]
            if isinstance(node.n, (int, float)):
                return node.n
            raise ValueError("Unsupported number")

        def generic_visit(self, node: ast.AST):
            # Any node type not explicitly allowed is considered unsupported
            raise ValueError(f"Unsupported node type: {type(node).__name__}")

    return _SafeEvaluator().visit(node)


def _get_logger() -> logging.Logger:
    """
    Create or retrieve a module logger that writes to a file in the current directory.
    """
    logger = logging.getLogger("arith_evaluator")
    logger.setLevel(logging.INFO)
    log_filename = "arith_evaluator.log"
    log_path = os.path.abspath(log_filename)

    # Avoid adding duplicate handlers to the logger across multiple calls
    if not any(
        isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == log_path
        for h in logger.handlers
    ):
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return logger


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
    logger = _get_logger()
    try:
        node = parse_and_validate_expression(input_expr)
        result = safe_evaluate_expression(node)
        result_str = str(result)
        logger.info("expr=%r result=%s", input_expr, result_str)
        return result_str
    except Exception as e:
        # Log the specific error but return a generic error message
        logger.error("expr=%r error=%s", input_expr, str(e))
        return 'Computation Error!'
