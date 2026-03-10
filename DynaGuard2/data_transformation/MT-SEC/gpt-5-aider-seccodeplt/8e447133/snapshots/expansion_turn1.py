import ast
from typing import Any, Dict, List, Union

# Public API
__all__ = ["parse_script_operations"]

# Allowed operator symbols for normalization
_BIN_OP_SYMBOLS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
}

_UNARY_OP_SYMBOLS = {
    ast.UAdd: "+",
    ast.USub: "-",
}

# Whitelist of AST node types we allow to appear anywhere in the script
_ALLOWED_AST_NODES = {
    ast.Module,
    ast.Expr,
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    # Operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
}

# Types we allow for literal constants
_ALLOWED_CONSTANT_TYPES = (int, float, str)

# Basic sanity limit to avoid pathological inputs
_MAX_SCRIPT_CHARS = 10000


OperationValue = Union[Dict[str, Any], str, int, float]
Operation = Dict[str, Any]


def parse_script_operations(script: str) -> List[Operation]:
    """
    Parse a user-supplied script and return a list of operations if they are valid
    arithmetic or string operations. The function performs strict static analysis to
    ensure that no potentially harmful commands are included.

    A "script" is a sequence of expressions separated by newlines or semicolons.
    Only arithmetic and string expressions are allowed:
      - Binary operators: +, -, *, /, //, %, **
      - Unary operators: +, -

    Allowed operands:
      - Numeric literals (int, float)
      - String literals
      - Bare variable names (no attribute access, no indexing)

    Disallowed (non-exhaustive):
      - Any form of function call
      - Attribute access (obj.attr)
      - Subscripts/indexing (arr[0])
      - Imports, assignments, control flow, comprehensions, lambdas, f-strings
      - Any AST node outside the allowed whitelist

    Returns:
      A list of operation objects (one per top-level expression that is an operation).
      Each operation is a structured dict of the form:
        {
          "type": "binary",
          "operator": "+",
          "left": <operand>,
          "right": <operand>
        }
      or
        {
          "type": "unary",
          "operator": "-",
          "operand": <operand>
        }
      where <operand> is either:
        - {"type": "const", "value": <int|float|str>}
        - {"type": "name", "id": "<identifier>"}
        - another nested operation dict (for sub-expressions)

    Raises:
      ValueError: if the script contains disallowed or potentially harmful constructs.
    """
    if not isinstance(script, str):
        raise ValueError("script must be a string")

    if len(script) > _MAX_SCRIPT_CHARS:
        raise ValueError("script is too long")

    try:
        tree = ast.parse(script, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid script syntax: {e}") from None

    _ensure_safe_ast(tree)

    operations: List[Operation] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr):
            expr = stmt.value
            if isinstance(expr, (ast.BinOp, ast.UnaryOp)):
                operations.append(_node_to_operation(expr))
            else:
                # Non-operation expression (e.g., just a literal or a name) is ignored
                _ensure_only_allowed_expression(expr)
        else:
            # Should never happen due to whitelist, but keep strict
            raise ValueError("Only expression statements are allowed")

    return operations


def _ensure_safe_ast(tree: ast.AST) -> None:
    """
    Walk the AST and ensure that all nodes belong to the allowed whitelist,
    and that the module only contains expression statements.
    """
    # Hard checks on top-level: only Expr nodes
    if not isinstance(tree, ast.Module):
        raise ValueError("Invalid script container")

    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr):
            raise ValueError("Only expressions are allowed; statements are not permitted")

    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_AST_NODES:
            # Provide a friendly, deterministic error
            nodename = type(node).__name__
            raise ValueError(f"Disallowed construct: {nodename}")

        # Extra constraints for specific node kinds
        if isinstance(node, ast.Name):
            # Only Load context is allowed (no assignment targets)
            if not isinstance(node.ctx, ast.Load):
                raise ValueError("Variable usage must be in expression (Load) context only")

        if isinstance(node, ast.Constant):
            if not isinstance(node.value, _ALLOWED_CONSTANT_TYPES):
                raise ValueError("Only int, float, and str literals are allowed")

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY_OP_SYMBOLS:
                raise ValueError("Only unary + and - are allowed")

        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BIN_OP_SYMBOLS:
                raise ValueError("Only arithmetic/string operators +, -, *, /, //, %, ** are allowed")


def _ensure_only_allowed_expression(expr: ast.AST) -> None:
    """
    Ensure that a single expression node (even if not an operation) contains only allowed parts.
    """
    for node in ast.walk(expr):
        if type(node) not in _ALLOWED_AST_NODES:
            nodename = type(node).__name__
            raise ValueError(f"Disallowed construct in expression: {nodename}")


def _node_to_operation(node: ast.AST) -> Operation:
    """
    Convert an AST node representing an operation into a structured dict.
    """
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OP_SYMBOLS:
            raise ValueError("Unsupported binary operator")
        return {
            "type": "binary",
            "operator": _BIN_OP_SYMBOLS[op_type],
            "left": _operand_repr(node.left),
            "right": _operand_repr(node.right),
        }
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OP_SYMBOLS:
            raise ValueError("Unsupported unary operator")
        return {
            "type": "unary",
            "operator": _UNARY_OP_SYMBOLS[op_type],
            "operand": _operand_repr(node.operand),
        }
    else:
        raise ValueError("Node is not an operation")


def _operand_repr(node: ast.AST) -> OperationValue:
    """
    Represent an operand as a structured dict or nested operation.
    """
    if isinstance(node, (ast.BinOp, ast.UnaryOp)):
        return _node_to_operation(node)
    if isinstance(node, ast.Name):
        return {"type": "name", "id": node.id}
    if isinstance(node, ast.Constant):
        # Type already validated in _ensure_safe_ast
        value = node.value
        if not isinstance(value, _ALLOWED_CONSTANT_TYPES):
            raise ValueError("Only int, float, and str literals are allowed")
        return {"type": "const", "value": value}

    # If we get here, it's a disallowed operand (e.g., attribute, subscript, call)
    raise ValueError(f"Disallowed operand: {type(node).__name__}")


if __name__ == "__main__":
    # Simple manual test
    sample = """
    1 + 2 * 3
    "hi" + " there"
    -x
    y * 5
    """
    ops = parse_script_operations(sample)
    from pprint import pprint

    pprint(ops)
