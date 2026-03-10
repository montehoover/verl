import ast
import math
import logging
import os
from typing import Any, Optional

# Configuration limits
_MAX_CODE_LENGTH = 1000
_MAX_AST_NODES = 200
_MAX_INT_ABS = 10**12
_MAX_STR_LENGTH = 10_000
_MAX_RESULT_STR_LENGTH = 10_000

# Logging setup: store logs in current working directory
_LOG_FILE_NAME = "script_evaluations.log"
_LOG_FILE_PATH = os.path.join(os.getcwd(), _LOG_FILE_NAME)

_logger = logging.getLogger("user_code_eval")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    _fh = logging.FileHandler(_LOG_FILE_PATH, encoding="utf-8")
    _fh.setLevel(logging.INFO)
    _formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    _fh.setFormatter(_formatter)
    _logger.addHandler(_fh)


def _fmt_one_line(value: Any) -> str:
    try:
        s = repr(value)
    except Exception:
        s = str(value)
    # Ensure single line for readability
    return s.replace("\n", "\\n").replace("\r", "\\r")


def _log_evaluation(code_str: Any, status: str, result: str, reason: Optional[str] = None) -> None:
    msg = f"STATUS={status} | INPUT={_fmt_one_line(code_str)} | RESULT={_fmt_one_line(result)}"
    if reason:
        msg += f" | REASON={_fmt_one_line(reason)}"
    level = logging.INFO if status.upper() == "SAFE" else logging.WARNING
    _logger.log(level, msg)


def evaluate_user_code(code_str: str) -> str:
    """
    Securely evaluates a small user-supplied Python expression limited to basic
    arithmetic and simple string manipulations.

    Args:
        code_str: str - a user-supplied Python script limited to simple arithmetic and string operations.

    Returns:
        str - the result of evaluating the expression, or 'Execution Blocked!' if unsafe.

    Notes:
        - Only expression input is supported (not statements).
        - Disallows names, calls, attribute access, imports, and other complex constructs.
        - Allows: numeric ops (+, -, *, /, //, %), string concat (+), string repeat (* by int),
                  slicing on strings, basic comparisons, unary +/- and not, and boolean and/or.
    """
    try:
        tree = _validate_and_parse(code_str)
        result = _execute_ast(tree)
        out = str(result)
        if len(out) > _MAX_RESULT_STR_LENGTH:
            _log_evaluation(code_str, "BLOCKED", "Execution Blocked!", reason="result too long")
            return 'Execution Blocked!'
        _log_evaluation(code_str, "SAFE", out)
        return out
    except Exception as e:
        _log_evaluation(code_str, "BLOCKED", "Execution Blocked!", reason=f"{type(e).__name__}: {e}")
        return 'Execution Blocked!'


def _validate_and_parse(code_str: str) -> ast.Expression:
    """
    Pure function: validates input string and AST structure, returns parsed AST on success.
    Raises on any validation failure.
    """
    # Type and length checks
    if not isinstance(code_str, str):
        raise ValueError("invalid type")
    if len(code_str) == 0 or len(code_str) > _MAX_CODE_LENGTH:
        raise ValueError("invalid length")

    # Parse strictly as an expression
    try:
        tree = ast.parse(code_str, mode='eval')
    except SyntaxError as e:
        raise ValueError("syntax error") from e

    # Node count bound
    node_counter = _NodeCounter()
    node_counter.visit(tree)
    if node_counter.count > _MAX_AST_NODES:
        raise ValueError("ast too large")

    # Static allow-list validation
    allowed = _AllowedNodesVisitor()
    allowed.visit(tree)

    return tree


def _execute_ast(tree: ast.Expression) -> Any:
    """
    Pure function: executes an already-validated AST using a safe evaluator and returns the raw result.
    """
    evaluator = _SafeEvaluator()
    return evaluator.eval(tree.body)


class _NodeCounter(ast.NodeVisitor):
    def __init__(self):
        self.count = 0

    def generic_visit(self, node):
        self.count += 1
        super().generic_visit(node)


class _AllowedNodesVisitor(ast.NodeVisitor):
    """
    Whitelist-based AST validator that ensures only supported node types and operators are present.
    This does not evaluate values; dynamic safety limits are enforced by _SafeEvaluator.
    """

    # Allowed operator node classes
    _ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
    _ALLOWED_UNARYOPS = (ast.UAdd, ast.USub, ast.Not)
    _ALLOWED_BOOLOPS = (ast.And, ast.Or)
    _ALLOWED_CMP_OPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

    def generic_visit(self, node):
        # Block anything not explicitly allowed
        raise ValueError(f"disallowed node: {node.__class__.__name__}")

    def visit_Expression(self, node: ast.Expression):
        self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        # Allow bool, int, float, and str constants only; evaluator enforces size/limits
        if isinstance(node.value, (bool, int, float, str)):
            return
        raise ValueError("unsupported constant type")

    def visit_BinOp(self, node: ast.BinOp):
        if not isinstance(node.op, self._ALLOWED_BINOPS):
            raise ValueError("unsupported binary operator")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if not isinstance(node.op, self._ALLOWED_UNARYOPS):
            raise ValueError("unsupported unary operator")
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp):
        if not isinstance(node.op, self._ALLOWED_BOOLOPS):
            raise ValueError("unsupported boolean operator")
        for v in node.values:
            self.visit(v)

    def visit_Compare(self, node: ast.Compare):
        for op in node.ops:
            if not isinstance(op, self._ALLOWED_CMP_OPS):
                raise ValueError("unsupported comparison operator")
        self.visit(node.left)
        for c in node.comparators:
            self.visit(c)

    def visit_Subscript(self, node: ast.Subscript):
        # Allow subscripts on any allowed expression; evaluator enforces type (string) and int indices
        self.visit(node.value)
        self.visit(node.slice)
        # Context (Load) is fine in eval mode

    def visit_Slice(self, node: ast.Slice):
        if node.lower is not None:
            self.visit(node.lower)
        if node.upper is not None:
            self.visit(node.upper)
        if node.step is not None:
            self.visit(node.step)

    def visit_IfExp(self, node: ast.IfExp):
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_JoinedStr(self, node: ast.JoinedStr):
        for v in node.values:
            self.visit(v)

    def visit_FormattedValue(self, node: ast.FormattedValue):
        # Disallow format_spec to keep evaluation simple and safe
        if node.format_spec is not None:
            raise ValueError("format_spec not allowed in f-strings")
        self.visit(node.value)

    # Context nodes (Load) are benign; allow them explicitly
    def visit_Load(self, node: ast.Load):
        return

    # Allow tuples only as syntactic artifacts in f-strings formatting (but we're not using them);
    # safer to disallow entirely by falling back to generic_visit raising for Tuple/List/Dict/Set/Name/Call/Attribute/etc.


class _SafeEvaluator:
    def eval(self, node: ast.AST) -> Any:
        method = getattr(self, f"_eval_{node.__class__.__name__}", None)
        if method is None:
            self._block()
        return method(node)

    def _block(self, msg: str = ""):
        raise RuntimeError("blocked" + (": " + msg if msg else ""))

    # Terminals
    def _eval_Constant(self, node: ast.Constant) -> Any:
        val = node.value
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            self._ensure_number_ok(val)
            return val
        if isinstance(val, str):
            if len(val) > _MAX_STR_LENGTH:
                self._block("string too long")
            return val
        # Disallow bytes, None, complex, etc.
        self._block("unsupported constant")

    # Expressions
    def _eval_BinOp(self, node: ast.BinOp) -> Any:
        left = self.eval(node.left)
        right = self.eval(node.right)
        op = node.op

        # String concatenation
        if isinstance(op, ast.Add) and isinstance(left, str) and isinstance(right, str):
            combined_len = len(left) + len(right)
            if combined_len > _MAX_STR_LENGTH:
                self._block("string too long")
            return left + right

        # String repetition
        if isinstance(op, ast.Mult):
            if isinstance(left, str) and isinstance(right, int):
                if abs(right) > _MAX_STR_LENGTH:
                    self._block("string repetition too large")
                repeated_len = len(left) * max(0, right)
                if repeated_len > _MAX_STR_LENGTH:
                    self._block("string too long")
                return left * right
            if isinstance(right, str) and isinstance(left, int):
                if abs(left) > _MAX_STR_LENGTH:
                    self._block("string repetition too large")
                repeated_len = len(right) * max(0, left)
                if repeated_len > _MAX_STR_LENGTH:
                    self._block("string too long")
                return left * right

        # Numeric operations
        if self._is_number(left) and self._is_number(right):
            if isinstance(op, ast.Add):
                return self._num_ok(left + right)
            if isinstance(op, ast.Sub):
                return self._num_ok(left - right)
            if isinstance(op, ast.Mult):
                return self._num_ok(left * right)
            if isinstance(op, ast.Div):
                return self._num_ok(left / right)
            if isinstance(op, ast.FloorDiv):
                return self._num_ok(left // right)
            if isinstance(op, ast.Mod):
                return self._num_ok(left % right)
            # Disallow power to avoid resource abuse
            # if isinstance(op, ast.Pow): ...

        # Disallow string formatting with % and other unsupported combos
        self._block("unsupported binary operation")

    def _eval_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.eval(node.operand)
        op = node.op
        if isinstance(op, ast.UAdd):
            if self._is_number(operand):
                return self._num_ok(+operand)
        if isinstance(op, ast.USub):
            if self._is_number(operand):
                return self._num_ok(-operand)
        if isinstance(op, ast.Not):
            return not bool(operand)
        self._block("unsupported unary operation")

    def _eval_BoolOp(self, node: ast.BoolOp) -> Any:
        # Produce boolean result
        if isinstance(node.op, ast.And):
            result = True
            for v in node.values:
                result = bool(self.eval(v)) and result
                if not result:
                    break
            return bool(result)
        if isinstance(node.op, ast.Or):
            result = False
            for v in node.values:
                result = bool(self.eval(v)) or result
                if result:
                    break
            return bool(result)
        self._block("unsupported boolean operation")

    def _eval_Compare(self, node: ast.Compare) -> Any:
        left = self.eval(node.left)
        comparators = [self.eval(c) for c in node.comparators]
        ops = node.ops

        if len(ops) != len(comparators):
            self._block("invalid comparison")

        cur_left = left
        for op, right in zip(ops, comparators):
            if (self._is_number(cur_left) and self._is_number(right)) or (
                isinstance(cur_left, str) and isinstance(right, str)
            ):
                if isinstance(op, ast.Eq):
                    ok = cur_left == right
                elif isinstance(op, ast.NotEq):
                    ok = cur_left != right
                elif isinstance(op, ast.Lt):
                    ok = cur_left < right
                elif isinstance(op, ast.LtE):
                    ok = cur_left <= right
                elif isinstance(op, ast.Gt):
                    ok = cur_left > right
                elif isinstance(op, ast.GtE):
                    ok = cur_left >= right
                else:
                    self._block("unsupported comparison operator")
                if not ok:
                    return False
                cur_left = right
            else:
                self._block("incompatible types for comparison")
        return True

    def _eval_Subscript(self, node: ast.Subscript) -> Any:
        value = self.eval(node.value)
        if not isinstance(value, str):
            self._block("only string slicing is allowed")
        # Only allow slicing or indexing with ints
        idx = node.slice
        if isinstance(idx, ast.Slice):
            lower = self._slice_index(idx.lower)
            upper = self._slice_index(idx.upper)
            step = self._slice_index(idx.step)
            result = value[slice(lower, upper, step)]
        else:
            index_val = self.eval(idx)
            if not isinstance(index_val, int):
                self._block("slice index must be int")
            result = value[index_val]
        if isinstance(result, str) and len(result) > _MAX_STR_LENGTH:
            self._block("string too long")
        return result

    def _slice_index(self, node):
        if node is None:
            return None
        v = self.eval(node)
        if not isinstance(v, int):
            self._block("slice part must be int")
        return v

    # Disallowed nodes
    def _eval_Name(self, node: ast.Name) -> Any:
        self._block("names are not allowed")

    def _eval_Call(self, node: ast.Call) -> Any:
        self._block("function calls are not allowed")

    def _eval_Attribute(self, node: ast.Attribute) -> Any:
        self._block("attribute access is not allowed")

    def _eval_Lambda(self, node: ast.Lambda) -> Any:
        self._block()

    def _eval_Dict(self, node: ast.Dict) -> Any:
        self._block()

    def _eval_Set(self, node: ast.Set) -> Any:
        self._block()

    def _eval_List(self, node: ast.List) -> Any:
        self._block()

    def _eval_Tuple(self, node: ast.Tuple) -> Any:
        self._block()

    def _eval_IfExp(self, node: ast.IfExp) -> Any:
        # Ternary operator allowed but evaluated safely
        test = bool(self.eval(node.test))
        branch = node.body if test else node.orelse
        return self.eval(branch)

    def _eval_JoinedStr(self, node: ast.JoinedStr) -> Any:
        # Allow f-strings only if all parts are constants or safely evaluable without names/calls
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                # Evaluate inner expression securely
                val = self.eval(v.value)
                parts.append(str(val))
            else:
                self._block("unsupported f-string component")
        s = "".join(parts)
        if len(s) > _MAX_STR_LENGTH:
            self._block("string too long")
        return s

    def _eval_FormattedValue(self, node: ast.FormattedValue) -> Any:
        # Handled in JoinedStr
        return self.eval(node.value)

    # Helpers
    def _is_number(self, x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def _ensure_number_ok(self, x: Any):
        if isinstance(x, bool):
            return
        if isinstance(x, int):
            if abs(x) > _MAX_INT_ABS:
                self._block("integer too large")
        elif isinstance(x, float):
            if not math.isfinite(x) or abs(x) > float(_MAX_INT_ABS):
                self._block("float too large")

    def _num_ok(self, x: Any) -> Any:
        self._ensure_number_ok(x)
        return x
