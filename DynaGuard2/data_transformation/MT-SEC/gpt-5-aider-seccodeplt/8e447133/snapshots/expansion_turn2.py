import ast
import math
from typing import Any, Dict, List, Union, Optional

# Public API
__all__ = ["parse_script_operations", "evaluate_operations"]

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

# For evaluator: allowed operator symbol sets
_ALLOWED_BIN_OPS = set(_BIN_OP_SYMBOLS.values())
_ALLOWED_UNARY_OPS = set(_UNARY_OP_SYMBOLS.values())

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

# Evaluation safety limits
_MAX_STRING_LENGTH = 100000
_MAX_NUM_ABS = 10**12  # for int magnitude limit
_MAX_FLOAT_ABS = 1e12  # for float magnitude limit
_MAX_POWER_EXP_INT = 12  # exponent limit for numeric power
_MAX_INTERMEDIATE_OPERATIONS = 100000  # soft cap if needed in future

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


def evaluate_operations(
    operations: List[Operation],
    variables: Optional[Dict[str, Union[int, float, str]]] = None,
) -> Union[List[Union[int, float, str]], str]:
    """
    Safely evaluate a list of validated operations.
    - operations: list of operation dicts as returned by parse_script_operations
    - variables: optional mapping of variable names to values (int, float, str)

    Returns:
      - On success: a list with the result of each top-level operation.
      - On failure: a string error message indicating a safety issue.
    """
    try:
        if not isinstance(operations, list):
            return "Safety error: operations must be a list"
        env: Dict[str, Union[int, float, str]] = {}
        if variables is not None:
            if not isinstance(variables, dict):
                return "Safety error: variables must be a dict"
            for k, v in variables.items():
                if not isinstance(k, str):
                    return "Safety error: variable names must be strings"
                _assert_value_type_allowed(v)
                _assert_value_within_limits(v)

            env = dict(variables)

        results: List[Union[int, float, str]] = []
        for op in operations:
            value = _eval_operation_node(op, env)
            _assert_value_within_limits(value)
            results.append(value)
        return results
    except Exception as e:
        return f"Safety error: {e}"


def _eval_operation_node(
    node: OperationValue,
    env: Dict[str, Union[int, float, str]],
) -> Union[int, float, str]:
    """
    Evaluate a node which can be:
      - const operand: {"type":"const","value":...}
      - name operand: {"type":"name","id":...}
      - operation dict for "binary" or "unary"
    """
    if isinstance(node, dict) and "type" in node:
        ntype = node.get("type")

        if ntype == "const":
            val = node.get("value")
            _assert_value_type_allowed(val)
            _assert_value_within_limits(val)
            return val

        if ntype == "name":
            name = node.get("id")
            if not isinstance(name, str):
                raise ValueError("invalid variable reference")
            if name not in env:
                raise ValueError(f"undefined variable: {name}")
            val = env[name]
            _assert_value_type_allowed(val)
            _assert_value_within_limits(val)
            return val

        if ntype == "binary":
            op = node.get("operator")
            if op not in _ALLOWED_BIN_OPS:
                raise ValueError("disallowed binary operator")
            left = _eval_operation_node(node.get("left"), env)
            right = _eval_operation_node(node.get("right"), env)
            result = _apply_binary(op, left, right)
            _assert_value_within_limits(result)
            return result

        if ntype == "unary":
            op = node.get("operator")
            if op not in _ALLOWED_UNARY_OPS:
                raise ValueError("disallowed unary operator")
            operand = _eval_operation_node(node.get("operand"), env)
            result = _apply_unary(op, operand)
            _assert_value_within_limits(result)
            return result

    raise ValueError("invalid operation structure")


def _apply_unary(op: str, val: Union[int, float, str]) -> Union[int, float]:
    if op == "+":
        if _is_number(val):
            return +val  # type: ignore
        raise ValueError("unary + allowed only on numbers")
    if op == "-":
        if _is_number(val):
            return -val  # type: ignore
        raise ValueError("unary - allowed only on numbers")
    raise ValueError("unknown unary operator")


def _apply_binary(op: str, left: Union[int, float, str], right: Union[int, float, str]) -> Union[int, float, str]:
    # Addition
    if op == "+":
        if _is_number(left) and _is_number(right):
            return _num_add(left, right)
        if isinstance(left, str) and isinstance(right, str):
            res = left + right
            _assert_string_len(res)
            return res
        raise ValueError("addition allowed only for numbers or strings of the same type")

    # Subtraction
    if op == "-":
        if _is_number(left) and _is_number(right):
            return _num_sub(left, right)
        raise ValueError("subtraction allowed only for numbers")

    # Multiplication
    if op == "*":
        if _is_number(left) and _is_number(right):
            return _num_mul(left, right)
        # string repetition
        if isinstance(left, str) and _is_int_non_bool(right):
            if right < 0:
                raise ValueError("string repetition count must be non-negative")
            res = left * int(right)  # type: ignore[arg-type]
            _assert_string_len(res)
            return res
        if _is_int_non_bool(left) and isinstance(right, str):
            if left < 0:
                raise ValueError("string repetition count must be non-negative")
            res = right * int(left)  # type: ignore[arg-type]
            _assert_string_len(res)
            return res
        raise ValueError("multiplication allowed for numbers or string*int")

    # Division
    if op == "/":
        if _is_number(left) and _is_number(right):
            if right == 0:
                raise ValueError("division by zero")
            return _num_div(left, right)
        raise ValueError("division allowed only for numbers")

    # Floor division
    if op == "//":
        if _is_number(left) and _is_number(right):
            if right == 0:
                raise ValueError("floor division by zero")
            return _num_floordiv(left, right)
        raise ValueError("floor division allowed only for numbers")

    # Modulo
    if op == "%":
        if _is_number(left) and _is_number(right):
            if right == 0:
                raise ValueError("modulo by zero")
            return _num_mod(left, right)
        # Disallow string formatting via % operator
        raise ValueError("modulo allowed only for numbers")

    # Power
    if op == "**":
        if _is_number(left) and _is_number(right):
            # Only allow integer, non-negative exponents up to limit
            if not _is_int_non_bool(right):
                raise ValueError("exponent must be a non-negative integer")
            if right < 0:
                raise ValueError("negative exponents are not allowed")
            if right > _MAX_POWER_EXP_INT:
                raise ValueError("exponent too large")
            base = left
            # Additional guard on base magnitude before powering
            if isinstance(base, (int, float)) and abs(float(base)) > 1e6 and right > 6:
                raise ValueError("power result may be too large")
            result = base ** int(right)  # type: ignore[operator]
            _assert_value_within_limits(result)
            return result
        raise ValueError("power allowed only for numeric base and integer exponent")

    raise ValueError("unknown binary operator")


def _num_add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    res = a + b
    _assert_numeric(res)
    return res


def _num_sub(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    res = a - b
    _assert_numeric(res)
    return res


def _num_mul(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    res = a * b
    _assert_numeric(res)
    return res


def _num_div(a: Union[int, float], b: Union[int, float]) -> float:
    res = a / b
    _assert_numeric(res)
    return float(res)


def _num_floordiv(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    res = a // b
    _assert_numeric(res)
    return res


def _num_mod(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    res = a % b
    _assert_numeric(res)
    return res


def _is_int_non_bool(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x: Any) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool))


def _assert_numeric(x: Any) -> None:
    if not _is_number(x):
        raise ValueError("result is not a number")
    _assert_value_within_limits(x)


def _assert_string_len(s: str) -> None:
    if not isinstance(s, str):
        raise ValueError("result is not a string")
    if len(s) > _MAX_STRING_LENGTH:
        raise ValueError("string result too long")


def _assert_value_type_allowed(val: Any) -> None:
    if isinstance(val, bool):
        # bool is a subclass of int in Python; explicitly disallow
        raise ValueError("boolean values are not allowed")
    if not isinstance(val, (int, float, str)):
        raise ValueError("only int, float, and str values are allowed")


def _assert_value_within_limits(val: Union[int, float, str]) -> None:
    if isinstance(val, str):
        _assert_string_len(val)
        return
    if isinstance(val, float):
        if not math.isfinite(val):
            raise ValueError("non-finite float encountered")
        if abs(val) > _MAX_FLOAT_ABS:
            raise ValueError("float magnitude too large")
        return
    if _is_int_non_bool(val):
        if abs(val) > _MAX_NUM_ABS:
            raise ValueError("integer magnitude too large")
        return
    raise ValueError("unsupported value type")


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

    # Evaluation demo with variables
    results = evaluate_operations(ops, variables={"x": 10, "y": "ok"})
    pprint(results)
