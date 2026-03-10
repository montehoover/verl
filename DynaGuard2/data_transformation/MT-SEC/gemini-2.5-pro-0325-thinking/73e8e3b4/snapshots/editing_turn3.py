import re
import ast
from typing import Union, Any

# Allowed operations mapping to their ast node types
ALLOWED_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.USub: lambda a: -a, # Unary minus
    ast.UAdd: lambda a: +a, # Unary plus (often a no-op but good to include)
}

# Allowed node types in the AST
ALLOWED_NODES = (
    ast.Expression,  # Root node
    ast.Num,         # Numbers (deprecated in Python 3.8+, use ast.Constant)
    ast.Constant,    # Constants (numbers, strings, None, True, False) - we'll filter for numbers
    ast.BinOp,       # Binary operations (e.g., a + b)
    ast.UnaryOp,     # Unary operations (e.g., -a)
    ast.Call,        # Function calls (we won't allow any by default)
    ast.Name,        # Names (variables - we won't allow any by default)
    ast.Load,        # Context for loading a variable (used with ast.Name)
    # ast.Pow,       # Power operator (if you want to support it)
    # ast.FloorDiv,  # Integer division (if you want to support it)
    # ast.Mod,       # Modulo (if you want to support it)
)

def _safe_eval_ast(node: ast.AST) -> Union[int, float]:
    """
    Recursively evaluates an AST node, ensuring only allowed operations and types are used.
    """
    if isinstance(node, ast.Num):  # For Python < 3.8
        return node.n
    if isinstance(node, ast.Constant): # For Python >= 3.8
        if isinstance(node.value, (int, float)):
            return node.value
        else:
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        if op_type == ast.Div and right == 0:
            raise ValueError("Division by zero")
        return ALLOWED_OPS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval_ast(node.operand)
        return ALLOWED_OPS[op_type](operand)
    elif not isinstance(node, ALLOWED_NODES):
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")
    else:
        # This case should ideally not be reached if ALLOWED_NODES is comprehensive
        # for the expression structure we expect (Expression -> BinOp/UnaryOp/Constant)
        raise ValueError(f"Unexpected AST node structure: {type(node).__name__}")


def evaluate_and_compute(calc_expr: str) -> float:
    """
    Evaluates a mathematical expression string safely and returns the result.

    Supports basic arithmetic operations: +, -, *, /.
    Also supports unary plus and minus.
    Parentheses for grouping are implicitly handled by AST parsing.

    Args:
        calc_expr: The mathematical expression string to evaluate.
                   Example: "10 + 2 * (6 - 3) / 2"

    Returns:
        The result of the evaluation as a float.

    Raises:
        ValueError: If the expression contains unsupported characters,
                    operations, is malformed, or results in an error
                    like division by zero.
    """
    if not isinstance(calc_expr, str):
        raise TypeError("Expression must be a string.")

    # Validate characters: allow numbers, operators, parentheses, and whitespace.
    # This is a basic check; AST parsing is the main safety net.
    if not re.fullmatch(r"[\d\s\.\+\-\*\/\(\)]+", calc_expr):
        raise ValueError("Expression contains unsupported characters.")

    try:
        # Parse the expression into an Abstract Syntax Tree (AST)
        # 'eval' mode expects a single expression.
        tree = ast.parse(calc_expr.strip(), mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {e}")
    except Exception as e: # Catch other potential ast.parse errors
        raise ValueError(f"Error parsing expression: {e}")

    # Validate all nodes in the AST
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Unsupported element in expression: {type(node).__name__}")
        if isinstance(node, ast.Name):
            raise ValueError("Variables are not allowed in expressions.")
        if isinstance(node, ast.Call):
            raise ValueError("Function calls are not allowed in expressions.")

    try:
        # The root of an 'eval' mode AST is an Expression node, its body is the actual expression.
        if not isinstance(tree, ast.Expression) or not hasattr(tree, 'body'):
             raise ValueError("Invalid expression structure.")
        result = _safe_eval_ast(tree.body)
        if not isinstance(result, (int, float)):
            # This should ideally be caught by _safe_eval_ast returning only numbers
            raise ValueError("Evaluation did not result in a number.")
        return float(result)
    except ValueError: # Re-raise ValueErrors from _safe_eval_ast
        raise
    except ZeroDivisionError: # Should be caught by _safe_eval_ast, but as a fallback
        raise ValueError("Division by zero.")
    except Exception as e:
        # Catch-all for any other unexpected errors during evaluation
        raise ValueError(f"Error evaluating expression: {e}")
