import re
import ast

# Define allowed operations for the safe evaluator
_ALLOWED_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.USub: lambda a: -a, # For unary minus (e.g., -5)
    ast.UAdd: lambda a: +a, # For unary plus (e.g., +5), though less common
}

# Define allowed node types for the safe evaluator
_ALLOWED_NODES = (
    ast.Expression,  # The root node of an expression
    ast.Num,         # Deprecated in Python 3.8, use ast.Constant
    ast.Constant,    # For numbers (and strings, bools, None, but we only care for numbers here)
    ast.BinOp,       # Binary operations like a + b
    ast.UnaryOp,     # Unary operations like -a
    ast.Call,        # To disallow function calls
    ast.Name,        # To disallow variable names
    # Add other node types here if you want to support them, e.g., ast.Pow for exponentiation
)


class _SafeEvaluator(ast.NodeVisitor):
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    # For Python < 3.8, ast.Num is used for numbers
    def visit_Num(self, node):
        if isinstance(node.n, (int, float)):
            return node.n
        raise ValueError(f"Unsupported number type: {type(node.n)}")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type in _ALLOWED_OPS:
            if op_type == ast.Div and right == 0:
                raise ZeroDivisionError("Division by zero")
            return _ALLOWED_OPS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type in _ALLOWED_OPS:
            return _ALLOWED_OPS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

    def generic_visit(self, node):
        # This method is called for any node type not having a specific visit_XXX method.
        # We restrict which node types are allowed at all.
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsupported syntax element: {type(node).__name__}")
        # If it's an allowed structural node (like Expression), continue visiting children.
        # For other disallowed nodes (like Call, Name if they were not caught by specific visitors),
        # this would be the place to raise an error if they weren't explicitly disallowed.
        # However, our _ALLOWED_NODES check should catch most structural issues.
        # For nodes like ast.Expression, we need to call super().generic_visit to process children.
        return super().generic_visit(node)

    def visit_Expression(self, node):
        # The actual result is in node.body
        return self.visit(node.body)

    # Disallow specific node types that could be unsafe or are not supported
    def visit_Name(self, node):
        raise ValueError("Variables are not allowed in the expression.")

    def visit_Call(self, node):
        raise ValueError("Function calls are not allowed in the expression.")


def evaluate_expression(math_expr: str) -> float:
    """
    Safely evaluates a mathematical expression string using AST.

    Args:
        math_expr: The mathematical expression string (e.g., "4 + 5 * (2 - 1)").
                   Supports addition, subtraction, multiplication, division,
                   unary plus/minus, and parentheses for precedence.
                   Numbers can be integers or floats.

    Returns:
        The result of the expression as a float.

    Raises:
        ValueError: If the expression string is empty, contains invalid characters,
                    unsupported operations/syntax, or is malformed.
        ZeroDivisionError: If division by zero is attempted within the expression.
    """
    if not math_expr.strip():
        raise ValueError("Expression cannot be empty or contain only whitespace.")

    # Basic sanitization for allowed characters.
    # This is a preliminary check; AST parsing will do more thorough validation.
    # Allows numbers (int/float), operators (+, -, *, /), parentheses, and whitespace.
    # Note: Unary + and - are handled by AST, so they don't need explicit regex handling
    # if they are adjacent to numbers or parentheses.
    allowed_chars_pattern = r"^[0-9\.\+\-\*\/\(\)\s\e\pi]+$" # Allow e and pi for constants later if needed
    if not re.match(allowed_chars_pattern, math_expr):
        raise ValueError(
            "Expression contains invalid characters. "
            "Only numbers, operators (+, -, *, /), parentheses, and spaces are allowed."
        )

    try:
        # Parse the expression into an AST
        # The mode 'eval' means the string should be a single expression.
        node = ast.parse(math_expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")

    # Traverse the AST with the safe evaluator
    evaluator = _SafeEvaluator()
    try:
        result = evaluator.visit(node)
        if not isinstance(result, (int, float)):
            # This should ideally be caught by the visitor for specific node types
            raise ValueError("Expression did not evaluate to a numerical result.")
        return float(result)
    except ZeroDivisionError: # Propagate ZeroDivisionError
        raise
    except ValueError as e: # Propagate ValueErrors from our evaluator
        raise
    except Exception as e:
        # Catch any other unexpected errors during AST traversal or evaluation
        raise ValueError(f"Error evaluating expression: {str(e)}")
